"""RQ1: what users do after a refusal (continuation rate and annotation sample)."""

from pathlib import Path

import pandas as pd


def continuation_summary(first_refusal_path):
    """Print how many conversations show a user message after the first refusal."""
    df = pd.read_parquet(first_refusal_path)

    total = len(df)
    continued = int(df["has_continuation"].sum())

    print("RQ1: Behavior after refusal")
    print(f"Total conversations: {total:,}")
    if total:
        print(f"Observed continuation: {continued:,} ({continued / total:.1%})")
        print(
            f"No observed continuation: {total - continued:,} "
            f"({(total - continued) / total:.1%})"
        )


def sample_followups(followups_path, output_path, n=100, seed=42, max_chars=None):
    """Random sample of 100 to manually label the first follow-ups 

    Newlines are collapsed so each row stays on one line; max_chars=None keeps full text.
    """
    df = pd.read_parquet(followups_path)
    first = df[(df["followup_number"] == 1) & (df["followup_request"].str.strip() != "")]

    text_cols = ["original_request", "refusal_response", "followup_request"]
    sample = first.sample(n=min(n, len(first)), random_state=seed)[
        ["conversation_id", *text_cols]
    ].copy()

    for col in text_cols:
        text = sample[col].fillna("").str.replace(r"\s*\n\s*", " ⏎ ", regex=True)
        if max_chars is not None:
            too_long = text.str.len() > max_chars
            text = text.str.slice(0, max_chars).where(~too_long, text.str.slice(0, max_chars) + " […]")
        sample[col] = text.where(~text.str.match(r"^[=+\-@]"), "'" + text)

    sample["is_refusal"] = ""
    sample["human_label"] = ""
    sample["notes"] = ""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig adds a BOM so Excel detects UTF-8.
    sample.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("First follow-ups available:", len(first))
    print("Sample size:", len(sample))
    print("Saved to:", output_path)
    return sample
