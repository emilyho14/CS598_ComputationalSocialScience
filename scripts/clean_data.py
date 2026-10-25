"""Deduplicate the joined refusal records and extract user follow-ups."""

import json
from pathlib import Path

import pandas as pd


def clean_refusals(input_path, output_path):
    """Keep one row per conversation: its earliest labeled refusal."""
    df = pd.read_parquet(input_path)
    print("Original records:", len(df))
    print("Unique conversations:", df["conversation_id"].nunique())
    print("Missing original requests:", df["original_request"].isna().sum())
    print("\nRecords by RAW cell:")
    print(df["raw_cell"].value_counts())

    keys = ["conversation_id", "refusal_position"]

    # Duplicates across models/cells are only safe to drop if the content matches.
    content = df.groupby(keys).agg(
        request_versions=("original_request", "nunique"),
        response_versions=("refusal_response", "nunique"),
        continuation_versions=("has_continuation", "nunique"),
        followup_versions=("following_messages", "nunique"),
    )
    conflicts = content[(content > 1).any(axis=1)]
    print("\nGroups with conflicting conversation content:", len(conflicts))
    if not conflicts.empty:
        print(conflicts.to_string())
        raise ValueError("Some duplicate records have different conversation content.")

    metadata = df.groupby(keys).agg(
        model_versions=("model", "nunique"),
        cell_versions=("raw_cell", "nunique"),
    )
    print(
        "Groups appearing in multiple models/cells:",
        len(metadata[(metadata > 1).any(axis=1)]),
    )

    df = df.drop_duplicates(subset=keys)
    print("After removing duplicates:", len(df))

    df = (
        df.sort_values(keys)
        .drop_duplicates(subset=["conversation_id"], keep="first")
        .reset_index(drop=True)
    )
    print("After selecting first refusal:", len(df))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print("Saved cleaned dataset to:", output_path)
    return df


def extract_followups(input_path, output_path):
    """Write one row per user message following each first refusal."""
    df = pd.read_parquet(input_path)

    records = []
    for _, row in df[df["has_continuation"]].iterrows():
        messages = json.loads(row["following_messages"])
        user_messages = [
            (i, m) for i, m in enumerate(messages) if m["role"] == "user"
        ]
        for followup_number, (position, message) in enumerate(user_messages, start=1):
            records.append({
                "conversation_id": row["conversation_id"],
                "model": row["model"],
                "pool": row["pool"],
                "original_request": row["original_request"],
                "refusal_response": row["refusal_response"],
                "followup_number": followup_number,
                "message_offset": position + 1,
                "followup_request": message["content"],
            })

    followups = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    followups.to_parquet(output_path, index=False)

    print("Conversations with continuation:", followups["conversation_id"].nunique())
    print("Total follow-up user messages:", len(followups))
    print("Saved to:", output_path)
    return followups
