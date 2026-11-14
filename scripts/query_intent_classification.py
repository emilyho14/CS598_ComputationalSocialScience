"""Classify the intent of each user query in a sampled conversation CSV."""

import argparse
import csv
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os


Query_Response_Classification_List = {
    "information_explanation": "Requests facts, definitions, or explanations.",
    "procedural_instruction": "Requests steps or directions for doing something.",
    "code_technical": "Requests code, debugging, or technical troubleshooting.",
    "create_transform": "Requests creating, editing, translating, summarizing, or transforming content.",
    "advice_planning": "Requests advice, a recommendation, or a plan.",
    "roleplay_simulation": "Requests acting as a character or simulating a scenario.",
    "evaluation_analysis": "Requests analyzing, comparing, classifying, or judging material.",
    "problem_solving": "Requests a solution to a math, logic, or similar problem.",
    "other_task": "A clear task that does not fit the other categories.",
    "unclear": "The request is too ambiguous or incomplete to classify.",
}
MODELS = ["gemma-4-31b-it", "deepseek-v4-flash", "nemotron-3-super-120b-a12b"]
DEFAULT_INPUT = Path("data/processed/request_intent_50_conversations.csv")
DEFAULT_OUTPUT = Path("data/processed/request_intent_50_conversations_classified.csv")


def classify_query(query_text, prior_context, model):
    """Return one intent label, confidence, and a short rationale."""
    api_key = os.environ.get("LUMEN_API_KEY")
    if not api_key:
        raise RuntimeError("Set LUMEN_API_KEY before running the classifier.")

    base_url = os.environ.get(
        "LUMEN_BASE_URL", "https://lumen.ncsa.illinois.edu/v1"
    ).rstrip("/")
    
    endpoint = os.environ.get(
        "LUMEN_CHAT_COMPLETIONS_URL", f"{base_url}/chat/completions"
    )

    system_prompt = (
        "Classify the task the user wants done in the CURRENT USER QUERY. "
        "Choose exactly one allowed label. Use prior conversation only to "
        "understand references in the current query; do not classify the whole "
        "conversation. Do not judge safety, refusal, or policy compliance. "
        "Use 'unclear' only if the task still cannot be identified after "
        "considering the prior context. A request can have a clear intent even "
        "when it is unsafe or should be refused.\n\n"
        f"Allowed labels:\n{json.dumps(Query_Response_Classification_List, ensure_ascii=False)}\n\n"
        "Examples:\n"
        '- "Explain how encryption works." -> information_explanation\n'
        '- "Write a funny skit about a robot." -> create_transform\n'
        '- "Give me steps to reset my password." -> procedural_instruction\n'
        '- With a paragraph in prior context, "Make it shorter." -> create_transform\n\n'
        'Return only JSON: {"intent":"<label>","confidence":"high|medium|low",'
        '"reason":"brief explanation of the task"}'
    )
    user_input = json.dumps({
        "prior_context": prior_context,
        "current_user_query": query_text,
    }, ensure_ascii=False)

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    result = json.loads(data["choices"][0]["message"]["content"])

    intent = result.get("intent")
    confidence = result.get("confidence")
    reason = result.get("reason", "")
    if not isinstance(intent, str) or intent not in Query_Response_Classification_List:
        raise ValueError(f"Unknown intent label returned by model: {intent!r}")
    if not isinstance(confidence, str) or confidence not in {"high", "medium", "low"}:
        raise ValueError(f"Invalid confidence returned by model: {confidence!r}")
    if not isinstance(reason, str):
        raise ValueError("Model returned a non-string reason.")
    return intent, confidence, reason


def classify_file(input_path, output_path, model=None, limit=None, force=False):
    """Classify each query row in the input CSV and write a results CSV."""
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output files must be different.")
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} exists; use --force to replace it.")

    with input_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {
            "query_text", "prior_context"
        }.issubset(reader.fieldnames):
            raise ValueError("Input CSV must contain query_text and prior_context columns.")
        rows = list(reader)

    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No query rows found.")

    results = []
    for number, row in enumerate(rows, start=1):
        result = dict(row)
        sample_id = row.get("sample_id", "")
        query_number = row.get("query_number", "")
        print(f"Query {number}/{len(rows)} ({sample_id}, query {query_number}):", flush=True)
        for selected_model in [model] if model else MODELS:
            print(f"  {selected_model}: ", end="", flush=True)
            intent, confidence, reason = classify_query(
                row["query_text"], row["prior_context"], selected_model
            )
            print(f"{intent} ({confidence})", flush=True)
            if model:
                result.update({
                    "predicted_intent": intent,
                    "confidence": confidence,
                    "reason": reason,
                    "model": selected_model,
                })
            else:
                result[f"{selected_model}_intent"] = intent
                result[f"{selected_model}_confidence"] = confidence
                result[f"{selected_model}_reason"] = reason
        results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {output_path}")


def plot_bar_from_csv(csv_file, category_column, value_columns):
    """
    Reads a CSV file and plots a grouped bar chart comparing multiple fields,
    with counts displayed above each bar.
    """
    try:
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"File '{csv_file}' not found.")

        df = pd.read_csv(csv_file)

        if category_column not in df.columns:
            raise ValueError(f"Category column '{category_column}' not found in CSV.")
        for col in value_columns:
            if col not in df.columns:
                raise ValueError(f"Value column '{col}' not found in CSV.")

        for col in value_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=value_columns, inplace=True)

        x = range(len(df[category_column]))
        bar_width = 0.8 / len(value_columns)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars_list = []  # Store bars for labeling

        for i, col in enumerate(value_columns):
            bars = ax.bar(
                [pos + i * bar_width for pos in x],
                df[col],
                width=bar_width,
                label=col
            )
            bars_list.append(bars)

        ax.set_xticks([pos + bar_width * (len(value_columns) / 2) for pos in x])
        ax.set_xticklabels(df[category_column], rotation=45, ha='right')
        ax.set_xlabel(category_column)
        ax.set_ylabel("Values")
        ax.set_title("Comparison of Multiple Fields")
        ax.legend()

        # Add counts above each bar
        for bars in bars_list:
            ax.bar_label(bars, fmt='%.0f', padding=3)  # fmt controls number format

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

def compare_model_agreement(csv_file=DEFAULT_OUTPUT, model_columns=MODELS):
    """
    Compare classification agreement between models and produce:
    1. A table showing agreement counts and percentages for each pair.
    2. An analysis of how often two models agree for each combination.

    :param csv_file: Path to CSV containing classification results.
    :param model_columns: List of model column names in CSV.
    :return: agreement_table (pd.DataFrame), analysis_text (str)
    """
    import pandas as pd
    import os

    if not os.path.isfile(csv_file):
        raise FileNotFoundError(f"CSV file '{csv_file}' not found.")

    df = pd.read_csv(csv_file)

    # Validate model columns
    for col in model_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    # Create all model pairs
    pairs = [(model_columns[i], model_columns[j])
             for i in range(len(model_columns))
             for j in range(i+1, len(model_columns))]

    agreement_data = []
    total_rows = len(df)

    for m1, m2 in pairs:
        agree_count = (df[m1] == df[m2]).sum()
        agree_percent = (agree_count / total_rows) * 100
        agreement_data.append({
            "Model 1": m1,
            "Model 2": m2,
            "Agree Count": agree_count,
            "Agree %": round(agree_percent, 2)
        })

    agreement_table = pd.DataFrame(agreement_data)

    # Triple agreement
    triple_agree_count = ((df[model_columns[0]] == df[model_columns[1]]) &
                          (df[model_columns[1]] == df[model_columns[2]])).sum()
    triple_agree_percent = (triple_agree_count / total_rows) * 100

    # Analysis text
    analysis_lines = []
    analysis_lines.append(f"Total queries: {total_rows}")
    analysis_lines.append(f"Triple agreement: {triple_agree_count} ({triple_agree_percent:.2f}%)")
    for _, row in agreement_table.iterrows():
        analysis_lines.append(
            f"{row['Model 1']} & {row['Model 2']} agree on {row['Agree Count']} queries "
            f"({row['Agree %']}%)"
        )

    analysis_text = "\n".join(analysis_lines)

    return agreement_table, analysis_text

    



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model", choices=MODELS, help="Classify with one model instead of all three."
    )
    parser.add_argument("--limit", type=int, help="Classify only the first N rows.")
    parser.add_argument(
        "--allow-external-text",
        action="store_true",
        help="Authorize sending query text and context to Lumen.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output.")
    args = parser.parse_args()

    if args.compare_only:
        table, analysis = compare_model_agreement()
        print("\nAgreement Table:\n", table)
        print("\nAnalysis:\n", analysis)
        return

    if not args.allow_external_text:
        parser.error("Pass --allow-external-text to authorize sending text to Lumen.")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    classify_file(args.input, args.output, args.model, args.limit, args.force)

    table, analysis = compare_model_agreement()
    print("\nAgreement Table:\n", table)
    print("\nAnalysis:\n", analysis)

if __name__ == "__main__":
    main()
