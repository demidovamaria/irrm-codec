import argparse
import json
import logging
import math
import inspect
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from irrm_codec.dataio import normalize_locus_name, read_airr_table


_thread_state = threading.local()


def _setup_logging(log_path=None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("irrm_codec.calc_pgen_1mm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute 1-mismatch pgen values for AIRR clonotypes using mirpy OLGA models."
    )
    parser.add_argument("--airr-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--output-stats-path")
    parser.add_argument("--locus")
    parser.add_argument("--chain", default="TRB")
    parser.add_argument("--species", default="human")
    parser.add_argument("--model-path")
    parser.add_argument("--clone-id-col", default="clone_id")
    parser.add_argument("--cdr3-col", default="junction_aa")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of sequences per saved chunk. Smaller chunks produce visible progress sooner.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--exact-pgen-col", default="pgen")
    parser.add_argument("--pgen-col", default="pgen_1mm")
    parser.add_argument("--exact-log10-pgen-col", default="log10_pgen")
    parser.add_argument("--log10-pgen-col", default="log10_pgen_1mm")
    parser.add_argument(
        "--is-d-present",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether the OLGA model contains a D segment. Default: infer from chain.",
    )
    parser.add_argument(
        "--mirpy-path",
        default=None,
        help="Optional path to a local mirpy checkout. Defaults to ../mirpy when present.",
    )
    parser.add_argument("--log-path")
    return parser.parse_args()


def _resolve_default_mirpy_path():
    candidate = Path(__file__).resolve().parents[1].parent / "mirpy"
    return candidate if candidate.exists() else None


def _import_olga_model_class(mirpy_path=None):
    try:
        from mir.basic.pgen import OlgaModel

        return OlgaModel
    except ModuleNotFoundError:
        resolved = Path(mirpy_path).resolve() if mirpy_path else _resolve_default_mirpy_path()
        if resolved is not None and resolved.exists():
            resolved_str = str(resolved)
            if resolved_str not in sys.path:
                sys.path.insert(0, resolved_str)
            from mir.basic.pgen import OlgaModel

            return OlgaModel
        raise ModuleNotFoundError(
            "Failed to import mir.basic.pgen.OlgaModel. "
            "Install mirpy-lib in the environment or pass --mirpy-path to a local checkout."
        )


def _parse_is_d_present(value):
    if value == "auto":
        return None
    return value == "true"


def _chunk_bounds(num_items, num_chunks=None, chunk_size=None):
    if num_items == 0:
        return []
    if chunk_size is not None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1.")
        return [
            (start, min(start + chunk_size, num_items))
            for start in range(0, num_items, chunk_size)
        ]
    if num_chunks is None:
        raise ValueError("Either num_chunks or chunk_size must be provided.")
    num_chunks = max(1, min(num_chunks, num_items))
    return [
        ((chunk_idx * num_items) // num_chunks, ((chunk_idx + 1) * num_items) // num_chunks)
        for chunk_idx in range(num_chunks)
        if (chunk_idx * num_items) // num_chunks < ((chunk_idx + 1) * num_items) // num_chunks
    ]


def _build_olga_model(OlgaModel, chain, species, model_path, is_d_present):
    signature = inspect.signature(OlgaModel.__init__)
    parameter_names = set(signature.parameters)

    if "locus" in parameter_names:
        kwargs = {"locus": chain.upper(), "species": species.lower()}
        if model_path:
            kwargs["model"] = model_path
        if is_d_present is not None:
            kwargs["is_d_present"] = is_d_present
        return OlgaModel(**kwargs)

    kwargs = {"chain": chain.upper()}
    if model_path:
        kwargs["model"] = model_path
    if is_d_present is not None:
        kwargs["is_d_present"] = is_d_present
    return OlgaModel(**kwargs)


def _compute_pgen_exact(model, seq):
    if hasattr(model, "compute_pgen_junction_aa"):
        return model.compute_pgen_junction_aa(seq)
    if hasattr(model, "compute_pgen_cdr3aa"):
        return model.compute_pgen_cdr3aa(seq)
    raise AttributeError(
        "OlgaModel does not provide a supported exact pgen method. "
        "Expected compute_pgen_junction_aa or compute_pgen_cdr3aa."
    )


def _compute_pgen_1mm(model, seq):
    if hasattr(model, "compute_pgen_junction_aa_1mm"):
        return model.compute_pgen_junction_aa_1mm(seq)
    if hasattr(model, "compute_pgen_cdr3aa_1mm"):
        return model.compute_pgen_cdr3aa_1mm(seq)
    raise AttributeError(
        "OlgaModel does not provide a supported 1mm pgen method. "
        "Expected compute_pgen_junction_aa_1mm or compute_pgen_cdr3aa_1mm."
    )


def _get_thread_model(chain, species, model_path, is_d_present, mirpy_path):
    cache_key = (chain.upper(), species.lower(), model_path, is_d_present, mirpy_path)
    model_cache = getattr(_thread_state, "model_cache", None)
    if model_cache is None:
        model_cache = {}
        _thread_state.model_cache = model_cache
    if cache_key not in model_cache:
        OlgaModel = _import_olga_model_class(mirpy_path)
        model_cache[cache_key] = _build_olga_model(
            OlgaModel=OlgaModel,
            chain=chain,
            species=species,
            model_path=model_path,
            is_d_present=is_d_present,
        )
    return model_cache[cache_key]


def _compute_chunk(
    chunk_id,
    start,
    end,
    sequences,
    chain,
    species,
    model_path,
    is_d_present,
    mirpy_path,
    batch_size,
):
    model = _get_thread_model(chain, species, model_path, is_d_present, mirpy_path)
    exact_values = np.empty(end - start, dtype=np.float64)
    one_mm_values = np.empty(end - start, dtype=np.float64)
    offset = 0
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        for seq in sequences[batch_start:batch_end]:
            exact_values[offset] = _compute_pgen_exact(model, seq)
            one_mm_values[offset] = _compute_pgen_1mm(model, seq)
            offset += 1
    return chunk_id, start, exact_values, one_mm_values


def _read_and_filter_airr(args):
    df = read_airr_table(args.airr_path, clone_id_col=args.clone_id_col)
    rows_before_locus = len(df)
    if args.locus is not None:
        locus = normalize_locus_name(args.locus)
        locus_series = df["locus"].astype(str).str.strip().str.lower().map(normalize_locus_name)
        df = df.loc[locus_series == locus].copy()
    else:
        locus = None

    if len(df) == 0:
        raise ValueError("AIRR table is empty after locus filtering.")
    if args.cdr3_col not in df.columns:
        raise ValueError(f"AIRR table does not contain required cdr3 column {args.cdr3_col!r}.")

    raw_cdr3 = df[args.cdr3_col]
    if raw_cdr3.isna().any():
        missing = int(raw_cdr3.isna().sum())
        raise ValueError(f"AIRR table contains {missing} missing values in {args.cdr3_col!r}.")

    sequences = raw_cdr3.astype(str).str.strip().str.upper()
    invalid_mask = sequences.eq("")
    if invalid_mask.any():
        raise ValueError(
            f"AIRR table contains {int(invalid_mask.sum())} empty CDR3 values in {args.cdr3_col!r}."
        )

    return df.reset_index(drop=True), sequences.tolist(), {
        "airr_path": str(Path(args.airr_path).resolve()),
        "rows_before_locus_filter": int(rows_before_locus),
        "rows_after_locus_filter": int(len(df)),
        "locus": locus,
        "chain": args.chain.upper(),
        "species": args.species.lower(),
        "cdr3_column": args.cdr3_col,
        "clone_id_column": args.clone_id_col,
    }


def _log10_or_neg_inf(value):
    if value > 0:
        return math.log10(value)
    return float("-inf")


def _save_table(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix in {".tsv", ".airr"}:
        df.to_csv(output_path, sep="\t", index=False)
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(
            f"Unsupported output extension {suffix!r}. Use .parquet, .tsv, .airr or .csv."
        )


def _save_stats(stats_path, payload):
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _chunk_store_dir(output_path):
    output_path = Path(output_path)
    return output_path.parent / f"{output_path.name}.chunks"


def _chunk_file_path(output_path, chunk_id):
    return _chunk_store_dir(output_path) / f"chunk_{chunk_id:04d}.tsv"


def _progress_file_path(output_path):
    return _chunk_store_dir(output_path) / "progress.json"


def _write_chunk_result(
    output_path,
    chunk_id,
    start,
    end,
    sequences,
    exact_values,
    one_mm_values,
    cdr3_col,
    exact_log10_pgen_col,
    log10_pgen_col,
):
    chunk_dir = _chunk_store_dir(output_path)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_df = pd.DataFrame(
        {
            cdr3_col: list(sequences[start:end]),
            exact_log10_pgen_col: np.array([_log10_or_neg_inf(value) for value in exact_values], dtype=np.float64),
            log10_pgen_col: np.array([_log10_or_neg_inf(value) for value in one_mm_values], dtype=np.float64),
        }
    )
    chunk_df.to_csv(_chunk_file_path(output_path, chunk_id), sep="\t", index=False)


def _load_existing_chunk(
    output_path,
    chunk_id,
    expected_start,
    expected_end,
    sequences,
    cdr3_col,
    exact_log10_pgen_col,
    log10_pgen_col,
):
    chunk_path = _chunk_file_path(output_path, chunk_id)
    if not chunk_path.exists():
        return None
    chunk_df = pd.read_csv(chunk_path, sep="\t")
    if list(chunk_df.columns) != [cdr3_col, exact_log10_pgen_col, log10_pgen_col]:
        raise ValueError(f"Unexpected chunk schema in {chunk_path}.")
    expected_len = expected_end - expected_start
    if len(chunk_df) != expected_len:
        raise ValueError(
            f"Chunk file {chunk_path} has {len(chunk_df)} rows, expected {expected_len}."
        )
    expected_sequences = np.asarray(sequences[expected_start:expected_end], dtype=object)
    observed_sequences = chunk_df[cdr3_col].astype(str).to_numpy(dtype=object)
    if not np.array_equal(observed_sequences, expected_sequences):
        raise ValueError(f"Chunk file {chunk_path} sequences do not match expected bounds.")
    return (
        chunk_df[exact_log10_pgen_col].to_numpy(dtype=np.float64),
        chunk_df[log10_pgen_col].to_numpy(dtype=np.float64),
    )


def _write_progress(output_path, payload):
    progress_path = _progress_file_path(output_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _assemble_from_chunks(
    df,
    output_path,
    cdr3_col,
    exact_pgen_col,
    pgen_col,
    exact_log10_pgen_col,
    log10_pgen_col,
    chunk_bounds,
    sequences,
):
    exact_log10_values = np.empty(len(df), dtype=np.float64)
    one_mm_log10_values = np.empty(len(df), dtype=np.float64)
    for chunk_id, (start, end) in enumerate(chunk_bounds):
        loaded = _load_existing_chunk(
            output_path,
            chunk_id,
            start,
            end,
            sequences,
            cdr3_col,
            exact_log10_pgen_col,
            log10_pgen_col,
        )
        if loaded is None:
            raise ValueError(f"Missing expected chunk {chunk_id} for final assembly.")
        exact_log_values, one_mm_log_values = loaded
        exact_log10_values[start:end] = exact_log_values
        one_mm_log10_values[start:end] = one_mm_log_values
    result_df = df.copy()
    result_df[exact_log10_pgen_col] = exact_log10_values
    result_df[log10_pgen_col] = one_mm_log10_values
    result_df[exact_pgen_col] = np.power(10.0, exact_log10_values)
    result_df[pgen_col] = np.power(10.0, one_mm_log10_values)
    return result_df


def main():
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be >= 1.")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    output_path = Path(args.output_path)
    default_log_path = output_path.with_suffix(output_path.suffix + ".log")
    logger = _setup_logging(args.log_path or default_log_path)
    logger.info("starting 1mm pgen calculation")

    df, sequences, input_stats = _read_and_filter_airr(args)
    num_rows = len(df)
    chunk_bounds = _chunk_bounds(num_rows, chunk_size=args.chunk_size)
    is_d_present = _parse_is_d_present(args.is_d_present)
    output_path = Path(args.output_path)
    completed_chunk_ids = []
    pending_chunks = []

    logger.info(
        "loaded rows=%d chain=%s locus=%s threads=%d chunk_size=%d batch_size=%d",
        num_rows,
        args.chain.upper(),
        input_stats["locus"],
        args.threads,
        args.chunk_size,
        args.batch_size,
    )

    for chunk_id, (start, end) in enumerate(chunk_bounds):
        existing_values = _load_existing_chunk(
            output_path,
            chunk_id,
            start,
            end,
            sequences,
            args.cdr3_col,
            args.exact_log10_pgen_col,
            args.log10_pgen_col,
        )
        if existing_values is not None:
            completed_chunk_ids.append(chunk_id)
            logger.info(
                "reused chunk=%d start=%d end=%d rows=%d",
                chunk_id,
                start,
                end,
                end - start,
            )
        else:
            pending_chunks.append((chunk_id, start, end))

    _write_progress(
        output_path,
        {
            **input_stats,
            "rows_total": int(num_rows),
            "threads_requested": int(args.threads),
            "chunk_size": int(args.chunk_size),
            "chunk_count": int(len(chunk_bounds)),
            "completed_chunks": completed_chunk_ids,
            "pending_chunks": [chunk_id for chunk_id, _start, _end in pending_chunks],
        },
    )

    with ThreadPoolExecutor(max_workers=min(len(pending_chunks), args.threads) or 1) as executor:
        futures = {
            executor.submit(
                _compute_chunk,
                chunk_id,
                start,
                end,
                sequences,
                args.chain,
                args.species,
                args.model_path,
                is_d_present,
                args.mirpy_path,
                args.batch_size,
            ): (chunk_id, start, end)
            for chunk_id, start, end in pending_chunks
        }
        for future in as_completed(futures):
            chunk_id, start, exact_values, one_mm_values = future.result()
            end = start + len(one_mm_values)
            _write_chunk_result(
                output_path,
                chunk_id,
                start,
                end,
                sequences,
                exact_values,
                one_mm_values,
                args.cdr3_col,
                args.exact_log10_pgen_col,
                args.log10_pgen_col,
            )
            completed_chunk_ids.append(chunk_id)
            _write_progress(
                output_path,
                {
                    **input_stats,
                    "rows_total": int(num_rows),
                    "threads_requested": int(args.threads),
                    "chunk_size": int(args.chunk_size),
                    "chunk_count": int(len(chunk_bounds)),
                    "completed_chunks": sorted(completed_chunk_ids),
                    "pending_chunks": sorted(
                        chunk_idx
                        for chunk_idx, _start, _end in pending_chunks
                        if chunk_idx not in completed_chunk_ids
                    ),
                },
            )
            logger.info(
                "completed chunk=%d start=%d end=%d rows=%d",
                chunk_id,
                start,
                end,
                len(one_mm_values),
            )

    result_df = _assemble_from_chunks(
        df,
        output_path,
        args.cdr3_col,
        args.exact_pgen_col,
        args.pgen_col,
        args.exact_log10_pgen_col,
        args.log10_pgen_col,
        chunk_bounds,
        sequences,
    )
    _save_table(result_df, output_path)

    stats_payload = {
        **input_stats,
        "output_path": str(output_path.resolve()),
        "stats_path": str(
            Path(args.output_stats_path).resolve()
            if args.output_stats_path
            else output_path.with_suffix(output_path.suffix + ".stats.json").resolve()
        ),
        "rows_written": int(len(result_df)),
        "threads": int(args.threads),
        "chunk_size": int(args.chunk_size),
        "batch_size": int(args.batch_size),
        "model_path": str(Path(args.model_path).resolve()) if args.model_path else None,
        "mirpy_path": str(Path(args.mirpy_path).resolve()) if args.mirpy_path else None,
        "is_d_present": is_d_present,
        "chunk_store_dir": str(_chunk_store_dir(output_path).resolve()),
        "chunk_count": int(len(chunk_bounds)),
        "exact_pgen_column": args.exact_pgen_col,
        "pgen_column": args.pgen_col,
        "exact_log10_pgen_column": args.exact_log10_pgen_col,
        "log10_pgen_column": args.log10_pgen_col,
    }
    stats_path = args.output_stats_path or output_path.with_suffix(output_path.suffix + ".stats.json")
    _save_stats(stats_path, stats_payload)
    logger.info("saved output=%s", output_path.resolve())
    logger.info("saved stats=%s", Path(stats_path).resolve())


if __name__ == "__main__":
    main()
