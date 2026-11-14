"""Build blinded, double-coding sheets for request-intent annotation.

The pilot samples first-refusal conversations separately by whether a user
follow-up was observed in the 10-message window. It includes the original
request and, when available, the first follow-up request. The sample is
stratified for annotation and is not a population-representative estimate.
"""

import argparse
from pathlib import Path

import pandas as pd


ANNOTATION_COLUMNS = [
    "intent_original",
    "intent_secondary_original",
    "domain_original",
    "framing_original",
    "certainty_original",
    "notes_original",
    "intent_followup",
    "intent_secondary_followup",
    "domain_followup",
    "framing_followup",
    "certainty_followup",
    "notes_followup",
]
TEXT_COLUMNS = ["original_request", "first_followup_request"]


def _spreadsheet_safe(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def build_sample(
    first_refusals_path,
    followups_path,
    output_dir,
    per_stratum=100,
    seed=42,
    force=False,
):
    """Write two identical-text, separate-coder CSV annotation sheets."""
    output_dir = Path(output_dir)
    coder_paths = [
        output_dir / "request_intent_coder_a.csv",
        output_dir / "request_intent_coder_b.csv",
    ]
    existing = [path for path in coder_paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Annotation sheets already exist; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
            + ". Use --force only if their annotations can be discarded."
        )
    if per_stratum < 1:
        raise ValueError("per_stratum must be at least 1.")

    conversations = pd.read_parquet(
        first_refusals_path,
        columns=[
            "conversation_id",
            "original_request",
            "has_continuation",
        ],
    )
    if conversations["conversation_id"].duplicated().any():
        raise ValueError("First-refusal input must contain one row per conversation.")

    first_followups = pd.read_parquet(followups_path)
    first_followups = first_followups[first_followups["followup_number"] == 1][
        ["conversation_id", "followup_request"]
    ].rename(columns={"followup_request": "first_followup_request"})
    if first_followups["conversation_id"].duplicated().any():
        raise ValueError("Follow-up input has multiple first follow-ups per conversation.")

    eligible = conversations.merge(
        first_followups, on="conversation_id", how="left", validate="one_to_one"
    )
    eligible["first_followup_request"] = eligible["first_followup_request"].fillna("")
    eligible["original_request"] = eligible["original_request"].fillna("")
    missing_request = eligible["original_request"].str.strip().eq("")
    excluded_missing_requests = int(missing_request.sum())
    eligible = eligible[~missing_request].copy()
    if (
        eligible.loc[eligible["has_continuation"], "first_followup_request"]
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError("A continuing conversation is missing its first follow-up.")

    sampled_groups = [
        group.sample(n=min(per_stratum, len(group)), random_state=seed)
        for _, group in eligible.groupby("has_continuation", sort=True)
    ]
    sampled = pd.concat(sampled_groups).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    sheet = sampled[
        ["conversation_id", "original_request", "first_followup_request"]
    ].copy()
    sheet.insert(
        0,
        "sample_id",
        [f"RI-{index:04d}" for index in range(1, len(sheet) + 1)],
    )
    for column in ANNOTATION_COLUMNS:
        sheet[column] = ""
    for column in TEXT_COLUMNS:
        sheet[column] = sheet[column].map(_spreadsheet_safe)

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in coder_paths:
        sheet.to_csv(path, index=False, encoding="utf-8-sig")

    counts = sampled["has_continuation"].value_counts().to_dict()
    print(f"Annotation cases: {len(sheet)}")
    print(
        "Strata sampled (continuation, no continuation): "
        f"{counts.get(True, 0)}, {counts.get(False, 0)}"
    )
    print(f"Excluded conversations with blank original requests: {excluded_missing_requests}")
    print(f"Each coder sheet: {len(sheet)} rows")
    for path in coder_paths:
        print("Saved:", path)
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--first-refusals",
        default="data/processed/first_refusal_conversations.parquet",
    )
    parser.add_argument(
        "--followups",
        default="data/processed/followup_requests.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/request_intent_annotation",
    )
    parser.add_argument("--per-stratum", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite coder sheets; existing annotations will be lost.",
    )
    args = parser.parse_args()
    build_sample(
        args.first_refusals,
        args.followups,
        args.output_dir,
        per_stratum=args.per_stratum,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
