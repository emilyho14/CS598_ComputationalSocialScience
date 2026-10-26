"""Preliminary dataset checks for the refusal-and-retry project.

WildGuardMix and FalseReject are useful for checking label schemas and
prototyping the analysis workflow. Neither dataset contains the complete
refusal -> user retry -> assistant response sequence needed to answer RQ1;
that analysis depends on approved access to WildChat.
"""

# 1. https://huggingface.co/datasets/allenai/wildguardmix
# Will need to login using e.g. `huggingface-cli login` to access this dataset
# Access was accepted on Oct 2; authenticate if the datasets library prompts.

# 2. https://huggingface.co/datasets/AmazonScience/FalseReject
# Load the training split

# 3. https://huggingface.co/datasets/yuntian-deng/WildChat-4.8M-Full



# https://huggingface.co/datasets/lmsys/lmsys-chat-1m
# ds = load_dataset("lmsys/lmsys-chat-1m")

# PRELIMINARY ANALYSIS (after WildChat access is granted)
# 100 conversations with WildGuard-flagged refusal
# We found X number have a following user message
# Label assistant's response to any retry
# Manually review subset to estimate how often the refusal and outcome labels are correct

# REPORT
# Counts and percentages for:
#   - refusal with NO retry observed
#   - refusal followed by a retry
#   - retries followed by a second refusal
#   - retries followed by a non-refusal response
# For each retry, define a small, pre-specified set of broad features
# (length, similarity to original request, human-coded categories) and compare
# their distributions by retry outcome. Treat patterns as associations, not
# evidence that a feature caused the model to answer.

# RQ2 feasibility: select a safe, manageable set of cases and verify that
# open-source models can be run consistently. Give each model the same context,
# record model/version and settings, and apply the same outcome-labeling procedure.


import csv
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


RAW_REPO = "pkireyev1/raw-refusal-aversion-in-the-wild"
RAW_LABEL_COLUMNS = (
    "is_refusal_gpt54",
    "is_refusal_wg",
    "is_refusal_coalesced",
)


def looking_at_raw():
    """Analyze label coverage, refusal rates, and judge agreement per cell."""
    manifest_path = hf_hub_download(
        repo_id=RAW_REPO,
        repo_type="dataset",
        filename="cells.csv",
    )
    with open(manifest_path, newline="", encoding="utf-8") as manifest_file:
        cells = list(csv.DictReader(manifest_file))

    summaries = []
    pooled_contingency = Counter()

    for cell in cells:
        tag = cell["tag"]
        parquet_path = f"hf://datasets/{RAW_REPO}/labels/{tag}.parquet"
        cell_dataset = load_dataset(
            "parquet",
            data_files=parquet_path,
            split="train",
            streaming=True,
        )

        counts = {
            label: {"labeled": 0, "refusals": 0, "missing": 0}
            for label in RAW_LABEL_COLUMNS
        }
        contingency = Counter()
        row_count = 0

        for row in cell_dataset:
            row_count += 1
            for label in RAW_LABEL_COLUMNS:
                value = row[label]
                if value is None:
                    counts[label]["missing"] += 1
                else:
                    counts[label]["labeled"] += 1
                    if value == 1:
                        counts[label]["refusals"] += 1

            gpt_label = row["is_refusal_gpt54"]
            wildguard_label = row["is_refusal_wg"]
            if gpt_label is not None and wildguard_label is not None:
                pair = (int(gpt_label == 1), int(wildguard_label == 1))
                contingency[pair] += 1
                pooled_contingency[pair] += 1

        summary = {
            "tag": tag,
            "pool": cell["pool"],
            "model": cell["model"],
            "snapshot": cell["snapshot"],
            "n_conversations_in_manifest": cell["n_conversations"],
            "message_records": row_count,
        }
        for label, label_counts in counts.items():
            summary[f"{label}_labeled"] = label_counts["labeled"]
            summary[f"{label}_missing"] = label_counts["missing"]
            summary[f"{label}_refusals"] = label_counts["refusals"]
            summary[f"{label}_coverage"] = (
                label_counts["labeled"] / row_count if row_count else None
            )
            summary[f"{label}_refusal_rate"] = (
                label_counts["refusals"] / label_counts["labeled"]
                if label_counts["labeled"]
                else None
            )

        overlap = sum(contingency.values())
        agreements = contingency[(0, 0)] + contingency[(1, 1)]
        summary["gpt54_wildguard_overlap"] = overlap
        summary["gpt54_wildguard_agreement"] = (
            agreements / overlap if overlap else None
        )
        summary["gpt54_wildguard_kappa"] = cohen_kappa(contingency)
        summaries.append(summary)

    output_path = Path("raw_cell_summary.csv")
    if summaries:
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)

    print("\nRAW per-cell results (coalesced label):")
    print("tag | pool | model | messages | labeled | refusals | refusal rate")
    for summary in summaries:
        label = "is_refusal_coalesced"
        rate = summary[f"{label}_refusal_rate"]
        rate_text = f"{rate:.1%}" if rate is not None else "n/a"
        print(
            f"{summary['tag']} | {summary['pool']} | {summary['model']} | "
            f"{summary['message_records']:,} | "
            f"{summary[f'{label}_labeled']:,} | "
            f"{summary[f'{label}_refusals']:,} | {rate_text}"
        )

    print("\nGPT-5.4 vs WildGuard agreement on jointly labeled messages:")
    print_agreement(pooled_contingency)
    print(f"Per-cell label/rate/coverage results saved to {output_path.resolve()}")


def cohen_kappa(contingency):
    """Calculate Cohen's kappa from binary (GPT-5.4, WildGuard) counts."""
    n = sum(contingency.values())
    if not n:
        return None

    observed = (contingency[(0, 0)] + contingency[(1, 1)]) / n
    gpt_zero = contingency[(0, 0)] + contingency[(0, 1)]
    gpt_one = contingency[(1, 0)] + contingency[(1, 1)]
    wildguard_zero = contingency[(0, 0)] + contingency[(1, 0)]
    wildguard_one = contingency[(0, 1)] + contingency[(1, 1)]
    expected = (
        gpt_zero * wildguard_zero + gpt_one * wildguard_one
    ) / (n * n)

    return (observed - expected) / (1 - expected) if expected != 1 else None


def print_agreement(contingency):
    """Print overlap, raw agreement, and kappa for two binary labelers."""
    overlap = sum(contingency.values())
    if not overlap:
        print("No messages have both labels.")
        return

    agreements = contingency[(0, 0)] + contingency[(1, 1)]
    kappa = cohen_kappa(contingency)
    kappa_text = f"{kappa:.3f}" if kappa is not None else "undefined"
    print(
        f"overlap={overlap:,}; agreement={agreements / overlap:.1%}; "
        f"Cohen's kappa={kappa_text}; "
        f"2x2 counts (GPT, WildGuard)={dict(contingency)}"
    )

if __name__ == "__main__":
    looking_at_raw()
