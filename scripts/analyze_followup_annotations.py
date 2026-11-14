"""Summarize manual labels in the follow-up annotation sample."""

import argparse
from pathlib import Path

import pandas as pd


BOOLEAN_LABELS = {
    "true": "refusal",
    "false": "not_refusal",
    "1": "refusal",
    "0": "not_refusal",
    "yes": "refusal",
    "no": "not_refusal",
}


def analyze_annotations(input_path, output_path):
    """Write category counts and refusal cross-tabs for an annotated CSV."""
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    required = {"human_label", "is_refusal"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing required columns: {sorted(missing)}")

    df = df[["human_label", "is_refusal"]].copy()
    df["human_label"] = (
        df["human_label"].fillna("unlabeled").astype(str).str.strip().str.lower()
    )
    df.loc[df["human_label"] == "", "human_label"] = "unlabeled"

    raw_refusal = df["is_refusal"].fillna("").astype(str).str.strip().str.lower()
    invalid = sorted(set(raw_refusal).difference(BOOLEAN_LABELS).difference({""}))
    if invalid:
        raise ValueError(f"Unrecognized is_refusal values: {invalid}")
    df["is_refusal"] = raw_refusal.map(BOOLEAN_LABELS).fillna("unlabeled")

    total = len(df)
    if total == 0:
        raise ValueError("The annotation CSV has no rows to summarize.")

    rows = []
    for label, count in df["human_label"].value_counts().sort_index().items():
        rows.append({
            "analysis": "followup_category",
            "human_label": label,
            "is_refusal": "",
            "count": int(count),
            "denominator": total,
            "percent": 100 * count / total,
        })

    for refusal, count in df["is_refusal"].value_counts().sort_index().items():
        rows.append({
            "analysis": "refusal_label",
            "human_label": "",
            "is_refusal": refusal,
            "count": int(count),
            "denominator": total,
            "percent": 100 * count / total,
        })

    cross_tab = pd.crosstab(df["human_label"], df["is_refusal"], dropna=False)
    for human_label in cross_tab.index:
        category_total = int(cross_tab.loc[human_label].sum())
        for refusal in cross_tab.columns:
            count = int(cross_tab.loc[human_label, refusal])
            rows.append({
                "analysis": "category_by_refusal",
                "human_label": human_label,
                "is_refusal": refusal,
                "count": count,
                "denominator": category_total,
                "percent": 100 * count / category_total if category_total else 0.0,
            })

    summary = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"Annotated rows: {total:,}")
    print("\nFollow-up category counts:")
    print(df["human_label"].value_counts().sort_index().to_string())
    print("\nRefusal annotation counts:")
    print(df["is_refusal"].value_counts().sort_index().to_string())
    print("\nFollow-up category by refusal annotation:")
    print(cross_tab.to_string())
    print(f"\nSaved summary to: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/processed/followup_annotation_sample.csv",
        help="Annotated follow-up CSV path",
    )
    parser.add_argument(
        "--output",
        default="data/processed/followup_annotation_summary.csv",
        help="Summary CSV output path",
    )
    args = parser.parse_args()
    analyze_annotations(args.input, args.output)


if __name__ == "__main__":
    main()