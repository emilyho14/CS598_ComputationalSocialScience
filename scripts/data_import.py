"""Download WildChat and join it with RAW's WildGuard refusal labels."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download

RAW_REPO = "pkireyev1/raw-refusal-aversion-in-the-wild"

JOINED_SCHEMA = pa.schema([
    ("conversation_id", pa.string()),
    ("model", pa.string()),
    ("pool", pa.string()),
    ("raw_cell", pa.string()),
    ("refusal_position", pa.int64()),
    ("original_request", pa.string()),
    ("refusal_response", pa.string()),
    ("following_messages", pa.string()),
    ("has_continuation", pa.bool_()),
])


def download_wildchat(wildchat_dir, batch_size=10_000):
    """Stream WildChat and save it locally as Parquet batches."""
    wildchat_dir = Path(wildchat_dir)
    wildchat_dir.mkdir(parents=True, exist_ok=True)

    stream = load_dataset(
        "yuntian-deng/WildChat-4.8M-Full", split="train", streaming=True
    )

    batch, file_number, total = [], 0, 0

    def flush():
        nonlocal batch, file_number, total
        output = wildchat_dir / f"wildchat_{file_number:04d}.parquet"
        Dataset.from_list(batch).to_parquet(str(output))
        total += len(batch)
        file_number += 1
        batch = []
        print(f"Saved {total:,} conversations")

    for conversation in stream:
        batch.append(conversation)
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()

    print(f"Finished! Saved {total:,} conversations to {wildchat_dir}")


def load_refusal_labels():
    """Return {conversation_id: [refusal info]} for WildGuard-labeled refusals."""
    manifest_path = hf_hub_download(
        repo_id=RAW_REPO, repo_type="dataset", filename="cells.csv"
    )
    with open(manifest_path, newline="", encoding="utf-8") as f:
        cells = list(csv.DictReader(f))

    refusals = defaultdict(list)
    for cell in cells:
        if not cell["pool"].startswith("wildchat"):
            continue

        print(f"Reading RAW cell: {cell['tag']}", flush=True)
        labels = load_dataset(
            "parquet",
            data_files=f"hf://datasets/{RAW_REPO}/labels/{cell['tag']}.parquet",
            split="train",
            streaming=True,
        )
        for row in labels:
            if row["is_refusal_wg"] == 1:
                refusals[row["conversation_id"]].append({
                    "message_number": int(row["message_number"]),
                    "model": cell["model"],
                    "pool": cell["pool"],
                    "raw_cell": cell["tag"],
                })

    print(f"\nFound {len(refusals):,} conversations with refusals")
    return refusals


def join_with_raw(wildchat_dir, output_path, window_size=10):
    """Match RAW refusal labels to WildChat and save one row per refusal."""
    files = sorted(Path(wildchat_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {wildchat_dir}")

    refusals = load_refusal_labels()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_matches = 0
    matched_ids = set()

    with pq.ParquetWriter(output_path, JOINED_SCHEMA) as writer:
        for file_index, file_path in enumerate(files, start=1):
            for batch in pq.ParquetFile(file_path).iter_batches(batch_size=1000):
                results = []
                for row in batch.to_pylist():
                    conversation_id = row["conversation_hash"]
                    if conversation_id not in refusals:
                        continue

                    messages = row["conversation"]
                    matched_ids.add(conversation_id)

                    for refusal in refusals[conversation_id]:
                        # RAW message numbers are 1-indexed.
                        index = refusal["message_number"] - 1
                        if not 0 <= index < len(messages):
                            continue
                        if messages[index]["role"] != "assistant":
                            continue

                        original_request = None
                        if index > 0 and messages[index - 1]["role"] == "user":
                            original_request = messages[index - 1]["content"]

                        following = messages[index + 1:index + 1 + window_size]

                        results.append({
                            "conversation_id": conversation_id,
                            "model": refusal["model"],
                            "pool": refusal["pool"],
                            "raw_cell": refusal["raw_cell"],
                            "refusal_position": refusal["message_number"],
                            "original_request": original_request,
                            "refusal_response": messages[index]["content"],
                            "following_messages": json.dumps(
                                following, ensure_ascii=False, default=str
                            ),
                            "has_continuation": any(
                                m["role"] == "user" for m in following
                            ),
                        })

                if results:
                    writer.write_table(
                        pa.Table.from_pylist(results, schema=JOINED_SCHEMA)
                    )
                    total_matches += len(results)

            print(
                f"[{file_index}/{len(files)}] {file_path.name} | "
                f"{total_matches:,} refusal records matched",
                flush=True,
            )

    print(f"\nMatched refusal records: {total_matches:,}")
    print(f"Matched conversation IDs: {len(matched_ids):,}")
    print(f"Unmatched conversation IDs: {len(refusals) - len(matched_ids):,}")
    print(f"Saved to: {output_path.resolve()}")
