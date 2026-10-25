# CS598 Computational Social Science: Refusals and Retries

What do users do after an LLM refuses their request, and which rephrasings are followed by an answer?

- **RQ1:** How do users rephrase after a refusal, and which rephrasings lead to an answer?
  - RQ1.1: What share of users give up, and what share try again?
  - RQ1.2: Of those who try again, how many are refused a second time?
  - RQ1.3: Which language features of a retry go with the model not refusing again?
- **RQ2:** Do these patterns hold for other models?

A refusal is an assistant message labeled as a refusal by WildGuard (`is_refusal_wg == 1` in RAW). "Give up" is operationalized as **no user message observed** after the refusal; the logs cannot show why a user stopped.

## Data sources

| Source                                                                                                                                            | Used for                                                | Access                |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | --------------------- |
| [WildChat-4.8M-Full](https://huggingface.co/datasets/allenai/WildChat-4.8M-Full) (`yuntian-deng/WildChat-4.8M-Full` is the copy the code streams) | Conversation text                                       | Gated on Hugging Face |
| [RAW](https://huggingface.co/datasets/pkireyev1/raw-refusal-aversion-in-the-wild) (Kireyev, 2026)                                                 | Refusal labels and join keys only; no conversation text | Public                |
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix), [FalseReject](https://huggingface.co/datasets/AmazonScience/FalseReject)    | Label-schema checks and background in `scripts/code.py` | WildGuardMix is gated |

RAW labels come from LLM judges (GPT-5.4-mini and WildGuard), not human annotators. Treat them as provisional until validated by manual review.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install datasets huggingface_hub pandas pyarrow jupyter
huggingface-cli login   # needed for the gated datasets
```

Run commands from the repository root; all paths are relative to it.

## Repository layout

```
scripts/
  code.py                          RAW label analysis (original preliminary analysis)
  data_import.py                   download WildChat, join with RAW labels
  clean_data.py                    deduplicate, keep first refusal, extract follow-ups
  rq1_post_refusal_behavior.py     RQ1 continuation summary and annotation sample
pipeline.ipynb                     runs the steps above; holds all file paths
raw_cell_summary.csv               output of scripts/code.py
data/
  wildchat/                        local WildChat copy (Parquet batches)
  processed/                       pipeline outputs
```

## Pipeline

Open `pipeline.ipynb` and run the cells in order. The first cell defines every file path.

### 1. Import data (`data_import.py`)

- `download_wildchat(WILDCHAT_DIR)` streams WildChat and saves it as Parquet batches of 10,000 conversations in `data/wildchat/`. This is large; the call is commented out in the notebook, so uncomment it only if the folder is empty.
- `join_with_raw(WILDCHAT_DIR, JOINED)`:
  1. Reads RAW's `cells.csv` and keeps only the WildChat cells (`pool` starts with `wildchat`).
  2. Streams each cell's `labels/{tag}.parquet` and collects messages where `is_refusal_wg == 1`.
  3. Matches `conversation_id` to WildChat's `conversation_hash`. RAW `message_number` is 1-indexed, so message `n` is `conversation[n - 1]`; it must be an assistant message.
  4. For each refusal, records the preceding user message (`original_request`), the refusal text, and the next 10 messages (`following_messages`, stored as JSON).
  5. Sets `has_continuation` to true if any of those next 10 messages is from the user.
- Output: `data/processed/refusal_conversations.parquet`, one row per refusal per RAW cell.

### 2. Clean data (`clean_data.py`)

- `clean_refusals(JOINED, FIRST_REFUSALS)`:
  - Prints record counts, missing original requests, and records per RAW cell.
  - The same conversation and position can appear in several RAW cells (different models). It raises an error if those duplicates disagree on content, then drops them.
  - Keeps only the **earliest** labeled refusal in each conversation.
  - Output: `data/processed/first_refusal_conversations.parquet`.
- `extract_followups(FIRST_REFUSALS, FOLLOWUPS)`:
  - Expands each conversation with a continuation into one row per user message within the 10-message window (`followup_number`, `followup_request`).
  - Output: `data/processed/followup_requests.parquet`.

### 3. RQ1 (`rq1_post_refusal_behavior.py`)

- `continuation_summary(FIRST_REFUSALS)` reports how many first-refusal conversations show a user message within the window and how many do not (RQ1.1).
- `sample_followups(FOLLOWUPS, ANNOTATION_SAMPLE)` draws up to 100 first follow-ups (seed 42) into `data/processed/followup_annotation_sample.csv`, with empty `human_label` and `notes` columns for manual coding.

## RAW label analysis (`scripts/code.py`)

Run separately: `python scripts/code.py`. It streams RAW's label files for all 19 model/pool cells and reports, per cell, record counts, label coverage, and refusal rates for the GPT-5.4-mini, WildGuard, and coalesced labels. It also compares GPT-5.4-mini and WildGuard on records where both labeled a message (agreement and Cohen's kappa). Results are saved to `raw_cell_summary.csv`. It also prints a FalseReject summary, which is background only.

## Limitations to keep in mind

- **Observation window:** a retry is only counted if it appears within 10 messages of the refusal. "No continuation" means none was observed in that window.
- **First refusal only:** later refusals in a conversation are not analyzed yet.
- **Label quality:** WildGuard and GPT-5.4-mini labels are not ground truth. Agreement between them is not accuracy, and the cells may contain different conversation samples, so per-model rates are descriptive, not a controlled comparison.
- **Associations, not causes:** retry features and outcomes are observational.
- **Sensitive content:** the data contains harmful or disturbing requests. Don't redistribute conversation text, and follow each dataset's license. WildChat and RAW's WildChat-derived files are ODC-BY; check terms for the others.

## Not done yet

- Manual labeling of `followup_annotation_sample.csv` (retry type) and a manual check of refusal labels.
- Labeling the assistant response to each retry (second refusal vs. answer) for RQ1.2.
- Retry language features and their association with outcomes (RQ1.3).
- RQ2: replaying the original request and retry on other open-source models.

## Reproducibility notes

- Add a `requirements.txt` with pinned versions once the environment settles.
- Add a `.gitignore` covering `.venv/`, `data/`, and `.DS_Store` so large data files aren't committed.
