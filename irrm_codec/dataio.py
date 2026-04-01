from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


AIRR_REQUIRED_COLUMNS = {"junction_aa", "v_call", "j_call", "locus"}


def read_airr_table(path, clone_id_col="clone_id"):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".tsv", ".airr"}:
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported AIRR file extension '{suffix}'. Use .tsv, .csv, .airr or .parquet."
        )

    missing_columns = AIRR_REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"AIRR table is missing required columns: {missing}")

    if clone_id_col in df.columns:
        if df[clone_id_col].isna().any():
            raise ValueError(f"AIRR table contains missing {clone_id_col} values.")
        if df[clone_id_col].duplicated().any():
            raise ValueError(f"AIRR table contains duplicate {clone_id_col} values.")

    return df.copy()


def extract_embedding_matrix(df, clone_id_col="clone_id", embedding_column="tcremp_emb"):
    if clone_id_col not in df.columns:
        raise ValueError(f"Embeddings table must contain '{clone_id_col}'.")
    if df[clone_id_col].isna().any():
        raise ValueError("Embeddings table contains missing clone_id values.")
    if df[clone_id_col].duplicated().any():
        raise ValueError("Embeddings table contains duplicate clone_id values.")

    if embedding_column in df.columns:
        matrix = np.stack(df[embedding_column].values).astype(np.float32)
        embedding_df = df[[clone_id_col]].copy()
    else:
        numeric_columns = [
            column
            for column in df.columns
            if column != clone_id_col and pd.api.types.is_numeric_dtype(df[column])
        ]
        if not numeric_columns:
            raise ValueError(
                f"Embeddings table must contain '{embedding_column}' or numeric embedding columns."
            )
        matrix = df[numeric_columns].to_numpy(dtype=np.float32)
        embedding_df = df[[clone_id_col]].copy()

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings matrix contains NaN or infinite values.")

    return embedding_df, matrix


def get_embedding_matrix(df, embedding_column="tcremp_emb"):
    if embedding_column in df.columns:
        matrix = np.stack(df[embedding_column].values).astype(np.float32)
    else:
        numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
        if not numeric_columns:
            raise ValueError(
                f"Embeddings table must contain '{embedding_column}' or numeric embedding columns."
            )
        matrix = df[numeric_columns].to_numpy(dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings matrix contains NaN or infinite values.")
    return matrix


def open_embeddings_parquet(path):
    return pq.ParquetFile(Path(path))


def get_embedding_columns(parquet_file, clone_id_col="clone_id", embedding_column="tcremp_emb"):
    schema = parquet_file.schema_arrow
    column_names = schema.names

    if embedding_column in column_names:
        return [embedding_column], True

    numeric_columns = []
    for field in schema:
        if field.name == clone_id_col:
            continue
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            numeric_columns.append(field.name)

    if not numeric_columns:
        raise ValueError(
            f"Embeddings table must contain '{embedding_column}' or numeric embedding columns."
        )
    return numeric_columns, False


def iter_embedding_batches(
    path,
    batch_size=4096,
    clone_id_col="clone_id",
    embedding_column="tcremp_emb",
    include_clone_id=True,
):
    parquet_file = open_embeddings_parquet(path)
    embedding_columns, uses_nested_embedding = get_embedding_columns(
        parquet_file,
        clone_id_col=clone_id_col,
        embedding_column=embedding_column,
    )
    requested_columns = list(embedding_columns)
    has_clone_id = clone_id_col in parquet_file.schema_arrow.names
    if include_clone_id and has_clone_id and clone_id_col not in requested_columns:
        requested_columns.insert(0, clone_id_col)

    for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=requested_columns):
        batch_df = record_batch.to_pandas()
        if uses_nested_embedding:
            matrix = np.stack(batch_df[embedding_column].values).astype(np.float32)
            clone_ids = batch_df[clone_id_col].tolist() if include_clone_id and has_clone_id else None
        else:
            matrix = batch_df[embedding_columns].to_numpy(dtype=np.float32, copy=False)
            clone_ids = batch_df[clone_id_col].tolist() if include_clone_id and has_clone_id else None

        yield clone_ids, matrix


def inspect_embeddings_file(path, clone_id_col="clone_id", embedding_column="tcremp_emb"):
    parquet_file = open_embeddings_parquet(path)
    embedding_columns, uses_nested_embedding = get_embedding_columns(
        parquet_file,
        clone_id_col=clone_id_col,
        embedding_column=embedding_column,
    )

    first_batch = next(
        iter_embedding_batches(
            path,
            batch_size=1,
            clone_id_col=clone_id_col,
            embedding_column=embedding_column,
            include_clone_id=False,
        ),
        None,
    )
    if first_batch is None:
        raise ValueError("Embeddings parquet file is empty.")

    _, first_matrix = first_batch
    return {
        "num_rows": int(parquet_file.metadata.num_rows),
        "embedding_dim": int(first_matrix.shape[1]),
        "has_clone_id": clone_id_col in parquet_file.schema_arrow.names,
        "uses_nested_embedding": uses_nested_embedding,
        "embedding_columns": embedding_columns,
    }


def load_airr_with_embeddings(
    airr_path,
    embeddings_path,
    locus=None,
    clone_id_col="clone_id",
    embedding_column="tcremp_emb",
):
    airr_df = read_airr_table(airr_path, clone_id_col=clone_id_col)
    if locus is not None:
        airr_df = airr_df[airr_df["locus"] == locus].reset_index(drop=True)
    if len(airr_df) == 0:
        raise ValueError("AIRR table is empty after locus filtering.")

    embeddings_raw = pd.read_parquet(embeddings_path)
    if clone_id_col not in airr_df.columns and len(airr_df) == len(embeddings_raw):
        merged = airr_df.copy().reset_index(drop=True)
        merged[clone_id_col] = np.arange(len(merged)).astype(str)
        emb = get_embedding_matrix(embeddings_raw, embedding_column=embedding_column)
        use_row_alignment = True
    else:
        embeddings_df, emb_matrix = extract_embedding_matrix(
            embeddings_raw,
            clone_id_col=clone_id_col,
            embedding_column=embedding_column,
        )
        embeddings_df["_embedding_index"] = np.arange(len(embeddings_df))

        merged = airr_df.merge(
            embeddings_df,
            left_on=clone_id_col,
            right_on=clone_id_col,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) == 0:
            raise ValueError(f"No rows matched between AIRR and embeddings tables by {clone_id_col}.")

        emb = emb_matrix[merged["_embedding_index"].to_numpy()]
        merged = merged.drop(columns=["_embedding_index"]).reset_index(drop=True)
        use_row_alignment = False

    stats = {
        "airr_rows": int(len(airr_df)),
        "embeddings_rows": int(len(embeddings_raw)),
        "merged_rows": int(len(merged)),
        "airr_unmatched_rows": int(len(airr_df) - len(merged)),
        "embeddings_unmatched_rows": int(len(embeddings_raw) - len(merged)),
        "clone_id_column": clone_id_col,
        "embedding_column": embedding_column if embedding_column in embeddings_raw.columns else None,
        "alignment_mode": "row_order" if use_row_alignment else "clone_id",
    }
    return merged, emb, stats
