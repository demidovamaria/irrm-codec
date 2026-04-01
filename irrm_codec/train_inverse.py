import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from irrm_codec.dataio import inspect_embeddings_file, iter_embedding_batches, read_airr_table
from irrm_codec.datasets import StreamingEmbeddingDataset, collate_inverse, validate_airr_dataframe
from irrm_codec.inverse_model import InverseModel
from irrm_codec.losses import inverse_loss, inverse_metrics
from irrm_codec.tokenization import decode
from irrm_codec.utils import (
    choose_device,
    move_to_device,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logging,
    split_indices,
    summarize_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the inverse IRRM-CODEC model.")
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--output-dir", default="artifacts/inverse")
    parser.add_argument("--locus", default="alpha")
    parser.add_argument("--clone-id-col", default="clone_id")
    parser.add_argument("--embedding-column", default="tcremp_emb")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--reader-batch-size", type=int, default=4096)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def load_airr_records(args):
    airr_df = read_airr_table(args.airr_path, clone_id_col=args.clone_id_col)
    if args.locus is not None:
        airr_df = airr_df[airr_df["locus"] == args.locus].reset_index(drop=True)
    if len(airr_df) == 0:
        raise ValueError("AIRR table is empty after locus filtering.")
    return airr_df


def build_embedding_iterator(args, alignment_mode):
    def iterator():
        row_offset = 0
        for clone_ids, emb_batch in iter_embedding_batches(
            args.embeddings_path,
            batch_size=args.reader_batch_size,
            clone_id_col=args.clone_id_col,
            embedding_column=args.embedding_column,
            include_clone_id=(alignment_mode == "clone_id"),
        ):
            if alignment_mode == "row_order":
                keys = list(range(row_offset, row_offset + len(emb_batch)))
                row_offset += len(emb_batch)
            else:
                keys = clone_ids
            yield keys, emb_batch

    return iterator


def prepare_streaming_splits(args):
    airr_df = load_airr_records(args)
    embedding_info = inspect_embeddings_file(
        args.embeddings_path,
        clone_id_col=args.clone_id_col,
        embedding_column=args.embedding_column,
    )
    records_by_clone = {
        row[args.clone_id_col]: {"junction_aa": row["junction_aa"]}
        for _, row in airr_df.iterrows()
        if args.clone_id_col in airr_df.columns
    }

    if args.clone_id_col not in airr_df.columns:
        if len(airr_df) != embedding_info["num_rows"]:
            raise ValueError(
                f"AIRR table has {len(airr_df)} rows but embeddings file has {embedding_info['num_rows']} rows. "
                "Row-order alignment is only supported when lengths match."
            )
        alignment_mode = "row_order"
        matched_keys = list(range(len(airr_df)))
        records_by_key = {idx: {"junction_aa": seq} for idx, seq in enumerate(airr_df["junction_aa"].tolist())}
    else:
        alignment_mode = "clone_id"
        seen_clone_ids = set()
        matched_keys = []
        for clone_ids, _emb_batch in iter_embedding_batches(
            args.embeddings_path,
            batch_size=args.reader_batch_size,
            clone_id_col=args.clone_id_col,
            embedding_column=args.embedding_column,
            include_clone_id=True,
        ):
            for clone_id in clone_ids:
                if clone_id in seen_clone_ids:
                    raise ValueError(f"Embeddings table contains duplicate {args.clone_id_col} values.")
                seen_clone_ids.add(clone_id)
                if clone_id in records_by_clone:
                    matched_keys.append(clone_id)

        if not matched_keys:
            raise ValueError(f"No rows matched between AIRR and embeddings tables by {args.clone_id_col}.")
        records_by_key = records_by_clone

    matched_key_set = set(matched_keys)
    if alignment_mode == "row_order":
        matched_df = airr_df.copy().reset_index(drop=True)
    else:
        matched_df = airr_df[airr_df[args.clone_id_col].isin(matched_key_set)].reset_index(drop=True)

    train_idx, val_idx, test_idx = split_indices(
        len(matched_keys),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    split_keys = {
        "train": {matched_keys[idx] for idx in train_idx},
        "val": {matched_keys[idx] for idx in val_idx},
        "test": {matched_keys[idx] for idx in test_idx},
    }
    merge_stats = {
        "airr_rows": int(len(airr_df)),
        "embeddings_rows": int(embedding_info["num_rows"]),
        "merged_rows": int(len(matched_keys)),
        "airr_unmatched_rows": int(len(airr_df) - len(matched_keys)),
        "embeddings_unmatched_rows": int(embedding_info["num_rows"] - len(matched_keys)),
        "clone_id_column": args.clone_id_col,
        "embedding_column": args.embedding_column,
        "alignment_mode": alignment_mode,
        "embedding_dim": int(embedding_info["embedding_dim"]),
    }
    return matched_df, records_by_key, split_keys, merge_stats, build_embedding_iterator(args, alignment_mode)


def fit_streaming_standardizer(iter_batches_fn, selected_keys, embedding_dim):
    count = 0
    feature_sum = np.zeros(embedding_dim, dtype=np.float64)
    feature_sum_sq = np.zeros(embedding_dim, dtype=np.float64)

    for keys, emb_batch in iter_batches_fn():
        row_indices = [row_idx for row_idx, key in enumerate(keys) if key in selected_keys]
        if not row_indices:
            continue

        embeddings = np.asarray(emb_batch[row_indices], dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embedding batch, got shape {embeddings.shape}.")
        if embeddings.shape[1] != embedding_dim:
            raise ValueError(f"Expected embedding dimension {embedding_dim}, got {embeddings.shape[1]}.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings matrix contains NaN or infinite values.")

        feature_sum += embeddings.sum(axis=0, dtype=np.float64)
        feature_sum_sq += np.square(embeddings, dtype=np.float64).sum(axis=0)
        count += embeddings.shape[0]

    if count == 0:
        raise ValueError("No training embeddings were found while fitting the standardizer.")
    if count != len(selected_keys):
        raise ValueError(
            f"Expected {len(selected_keys)} training embeddings while fitting the standardizer, found {count}."
        )

    mean = feature_sum / count
    variance = np.maximum(feature_sum_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build_dataloader(
    records_by_key,
    selected_keys,
    iter_batches_fn,
    batch_size,
    max_len,
    shuffle,
    num_workers,
    mean,
    std,
    seed,
):
    dataset = StreamingEmbeddingDataset(
        task="inverse",
        records_by_key=records_by_key,
        selected_keys=selected_keys,
        iter_embedding_batches_fn=iter_batches_fn,
        max_len=max_len,
        mean=mean,
        std=std,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_inverse,
    )


def exact_match_rate(pred_tokens, target_tokens):
    exact_matches = 0
    total = pred_tokens.size(0)
    for pred_row, target_row in zip(pred_tokens.tolist(), target_tokens.tolist()):
        if decode(pred_row) == decode(target_row):
            exact_matches += 1
    return exact_matches / max(total, 1)


def run_epoch(model, loader, optimizer, device, stage, epoch, num_epochs, log_interval, show_progress):
    is_train = optimizer is not None
    model.train(mode=is_train)
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(epoch)

    metric_sums = {
        "loss": 0.0,
        "token_accuracy": 0.0,
        "length_accuracy": 0.0,
        "exact_match": 0.0,
        "unk_fraction": 0.0,
    }
    steps = 0
    total_steps = len(loader)
    progress = tqdm(
        loader,
        total=total_steps,
        desc=f"{stage} {epoch}/{num_epochs}",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    )

    for step, batch in enumerate(progress, start=1):
        emb, decoder_input, target, lengths, unk_fraction = move_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            logits, length_logits = model(emb, decoder_input)
            loss = inverse_loss(logits, target, length_logits, lengths)
            metrics = inverse_metrics(logits, target, length_logits, lengths)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            exact_match = 0.0
        else:
            pred_tokens, _predicted_lengths = model.generate(emb, max_len=model.max_len)
            exact_match = exact_match_rate(pred_tokens, target)

        metric_sums["loss"] += loss.item()
        metric_sums["token_accuracy"] += metrics["token_accuracy"]
        metric_sums["length_accuracy"] += metrics["length_accuracy"]
        metric_sums["exact_match"] += exact_match
        metric_sums["unk_fraction"] += float(unk_fraction.item())
        steps += 1

        should_update = step == total_steps or (log_interval > 0 and step % log_interval == 0)
        if should_update and show_progress:
            avg_metrics = summarize_metrics(metric_sums, steps)
            progress.set_postfix(
                loss=f"{avg_metrics['loss']:.4f}",
                tok_acc=f"{avg_metrics['token_accuracy']:.4f}",
                len_acc=f"{avg_metrics['length_accuracy']:.4f}",
                exact=f"{avg_metrics['exact_match']:.4f}",
            )

    if show_progress:
        progress.close()
    return summarize_metrics(metric_sums, steps)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir / "train.log")

    logger.info("starting inverse training")
    logger.info("output_dir=%s", output_dir.resolve())
    logger.info("device=%s seed=%d", device, args.seed)
    logger.info(
        "hyperparameters batch_size=%d reader_batch_size=%d epochs=%d lr=%.6f weight_decay=%.6f max_len=%d num_workers=%d log_interval=%d",
        args.batch_size,
        args.reader_batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.max_len,
        args.num_workers,
        args.log_interval,
    )

    df, records_by_key, split_keys, merge_stats, iter_batches_fn = prepare_streaming_splits(args)
    data_stats = validate_airr_dataframe(df, max_len=args.max_len)
    data_stats["embedding_dim"] = merge_stats["embedding_dim"]

    mean, std = fit_streaming_standardizer(
        iter_batches_fn,
        split_keys["train"],
        merge_stats["embedding_dim"],
    )

    train_loader = build_dataloader(
        records_by_key,
        split_keys["train"],
        iter_batches_fn,
        args.batch_size,
        args.max_len,
        True,
        args.num_workers,
        mean,
        std,
        args.seed,
    )
    val_loader = build_dataloader(
        records_by_key,
        split_keys["val"],
        iter_batches_fn,
        args.batch_size,
        args.max_len,
        False,
        args.num_workers,
        mean,
        std,
        args.seed,
    )
    test_loader = build_dataloader(
        records_by_key,
        split_keys["test"],
        iter_batches_fn,
        args.batch_size,
        args.max_len,
        False,
        args.num_workers,
        mean,
        std,
        args.seed,
    )

    model = InverseModel(embedding_dim=merge_stats["embedding_dim"], max_len=args.max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_parameters = sum(param.numel() for param in model.parameters())
    num_trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)

    logger.info(
        "loaded data total=%d train=%d val=%d test=%d embedding_dim=%d",
        len(df),
        len(split_keys["train"]),
        len(split_keys["val"]),
        len(split_keys["test"]),
        merge_stats["embedding_dim"],
    )
    logger.info(
        "dataloader batches train=%d val=%d test=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    logger.info(
        "model parameters total=%d trainable=%d",
        num_parameters,
        num_trainable_parameters,
    )

    save_json(
        output_dir / "data_stats.json",
        {
            **data_stats,
            **merge_stats,
            "airr_path": args.airr_path,
            "embeddings_path": args.embeddings_path,
            "train_size": int(len(split_keys["train"])),
            "val_size": int(len(split_keys["val"])),
            "test_size": int(len(split_keys["test"])),
            "standardizer": {"mean_path": "mean.npy", "std_path": "std.npy"},
            "checkpoints": {"best": "best.pt", "last": "last.pt"},
        },
    )
    np.save(output_dir / "mean.npy", mean)
    np.save(output_dir / "std.npy", std)

    best_val_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        logger.info("epoch %d/%d started", epoch, args.epochs)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            "train",
            epoch,
            args.epochs,
            args.log_interval,
            not args.no_progress,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            "val",
            epoch,
            args.epochs,
            args.log_interval,
            not args.no_progress,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            extra={"task": "inverse", "max_len": args.max_len, "embedding_dim": merge_stats["embedding_dim"]},
        )
        logger.info("saved checkpoint path=%s", output_dir / "last.pt")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                extra={"task": "inverse", "max_len": args.max_len, "embedding_dim": merge_stats["embedding_dim"]},
            )
            logger.info("new best checkpoint path=%s val_loss=%.4f", output_dir / "best.pt", best_val_loss)

        logger.info(
            "epoch=%d summary train_loss=%.4f train_tok_acc=%.4f train_len_acc=%.4f val_loss=%.4f val_tok_acc=%.4f val_len_acc=%.4f val_exact=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["token_accuracy"],
            train_metrics["length_accuracy"],
            val_metrics["loss"],
            val_metrics["token_accuracy"],
            val_metrics["length_accuracy"],
            val_metrics["exact_match"],
        )

    test_metrics = run_epoch(
        model,
        test_loader,
        None,
        device,
        "test",
        args.epochs,
        args.epochs,
        args.log_interval,
        not args.no_progress,
    )
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "test_metrics.json", test_metrics)
    logger.info(
        "test summary loss=%.4f tok_acc=%.4f len_acc=%.4f exact=%.4f unk=%.4f",
        test_metrics["loss"],
        test_metrics["token_accuracy"],
        test_metrics["length_accuracy"],
        test_metrics["exact_match"],
        test_metrics["unk_fraction"],
    )


if __name__ == "__main__":
    main()
