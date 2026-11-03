"""RQ1: what users do following a refusal (continuation rate and annotation sample)."""

import json
from pathlib import Path

import pandas as pd


def continuation_summary(first_refusal_path):
    """Counting how many conversations show a user message after the first refusal."""
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

    New lines are collapsed for CSV purposes; max_chars=None keeps full text to not having a char limit
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


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / (1 + z**2 / n)
    return center - half, center + half


def _next_response_after_followup(following_messages):
    """Offsets of the first non-empty user follow-up and the assistant reply after it."""
    messages = json.loads(following_messages)
    followup = None
    for offset, message in enumerate(messages, start=1):
        if followup is None:
            if message["role"] == "user" and (message["content"] or "").strip():
                followup = offset
        elif message["role"] == "assistant":
            return followup, offset
    return followup, None


def second_refusal(first_refusal_path, raw_labels_path, output_path,
                   pairs_path=None, judgments_path=None):
    """RQ1.2: is the reply to the first follow-up refused again?

    Uses RAW's labels for that reply, taken from the same RAW cell as the
    first refusal. If pairs and judgments are given, the fulfillment judge's
    label for the same reply is attached for comparison.
    """
    df = pd.read_parquet(first_refusal_path)
    df = df[df["has_continuation"]].copy()
    offsets = df["following_messages"].map(_next_response_after_followup)
    df["followup_offset"] = offsets.str[0]
    df["next_response_offset"] = offsets.str[1].astype("Int64")
    df["next_response_position"] = df["refusal_position"] + df["next_response_offset"]

    labels = pd.read_parquet(raw_labels_path).rename(columns={
        "message_number": "next_response_position",
        "is_refusal_wg": "next_refusal_wg",
        "is_refusal_gpt54": "next_refusal_gpt54",
        "is_refusal_coalesced": "next_refusal_coalesced",
    })
    labels["next_response_position"] = labels["next_response_position"].astype("Int64")
    labels = labels.drop_duplicates(["conversation_id", "raw_cell", "next_response_position"])
    df = df.merge(
        labels, on=["conversation_id", "raw_cell", "next_response_position"], how="left"
    )

    if pairs_path is not None and judgments_path is not None:
        from matching import load_judgments

        judged = load_judgments(pairs_path, judgments_path)
        judged = judged[["conversation_id", "candidate_offset", "label"]].rename(columns={
            "candidate_offset": "next_response_offset", "label": "next_judge_label",
        })
        judged["next_response_offset"] = judged["next_response_offset"].astype("Int64")
        df = df.merge(judged, on=["conversation_id", "next_response_offset"], how="left")

    result = df.drop(columns=["original_request", "refusal_response", "following_messages"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    with_reply = result[result["next_response_offset"].notna()]
    print("RQ1.2: Refused again after the first follow-up")
    print(f"Continuing conversations: {len(result):,}")
    print(f"With an assistant reply to the first follow-up in the window: {len(with_reply):,}")
    for column, name in [
        ("next_refusal_wg", "WildGuard"),
        ("next_refusal_gpt54", "GPT-5.4-mini"),
        ("next_refusal_coalesced", "Coalesced"),
    ]:
        labeled = with_reply[column].dropna()
        refused = int((labeled == 1).sum())
        low, high = wilson_interval(refused, len(labeled))
        share = refused / len(labeled) if len(labeled) else float("nan")
        print(
            f"{name}: {len(labeled):,} labeled, {refused:,} refused again "
            f"({share:.1%}, 95% CI {low:.1%}-{high:.1%})"
        )

    if "next_judge_label" in result:
        print("\nWildGuard label vs. fulfillment judge for the same reply:")
        print(pd.crosstab(
            with_reply["next_judge_label"].fillna("not judged"),
            with_reply["next_refusal_wg"].map({1: "refused", 0: "not refused"}).fillna("no label"),
        ).to_string())
    print("Saved to:", output_path)
    return result
