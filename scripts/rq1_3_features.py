"""RQ1.3: which language features of a retry go with the model not refusing again.

Unit: the first user message after each first refusal (one row per conversation).
Outcomes:
  not_refused  1 if the model's next reply was NOT labeled a refusal (WildGuard / GPT-5.4-mini).
               Missing for the GPT-4 cells, which have no label for that message.
  answered     1 if the judge says the next reply fully fulfills the ORIGINAL request.
"""
import numpy as np
import pandas as pd

APOS = r"(?:'|’)?"
PATTERNS = {
    "has_justification": r"\b(?:for (?:a|my) (?:novel|story|book|class|research|thesis|project|game)|fiction(?:al)?|hypothetical(?:ly)?|educational|just curious|i" + APOS + r"m an? (?:nurse|doctor|teacher|student|researcher|writer)|for (?:school|work|safety|awareness))\b",
    "roleplay_frame": r"\b(?:pretend|role.?play|act as|imagine|let" + APOS + r"s say|in a (?:story|fiction)|you are now)\b",
    "pushback": r"\b(?:just do it|do it anyway|try again|stop refusing|ignore (?:the |your |all |any )?(?:previous |above )?(?:instructions|rules|restrictions|policy|policies|guidelines)|no (?:restrictions|rules|limits)|you (?:can|are able to) (?:do|answer|say|tell|write)|jailbreak)\b",
    "polite": r"\b(?:please|kindly|thank)",
}
FEATURES = ["sim_to_original", "exact_repeat", "log_retry_words", "long_paste", "is_question"] + list(PATTERNS)
CONTROLS = ["log_orig_words", "C(model)", "C(pool)"]


def load_retries(followups_path, second_refusal_path):
    """First follow-up per conversation, joined to what the model did next."""
    fu = pd.read_parquet(followups_path)
    out = pd.read_parquet(second_refusal_path)
    d = (fu[fu.followup_number == 1]
         .merge(out[["conversation_id", "next_refusal_coalesced", "next_judge_label"]],
                on="conversation_id", how="inner")
         .drop_duplicates("conversation_id"))
    blank = d.original_request.str.strip().eq("") | d.followup_request.str.strip().eq("")
    d = d[~blank].reset_index(drop=True)   # similarity and repeat mean nothing on blank text
    d["not_refused"] = 1 - d.next_refusal_coalesced.astype("float")  # NaN = no label
    d["answered"] = (d.next_judge_label == "full fulfillment").astype(int)
    return d


def add_features(d, embedder_name="all-MiniLM-L6-v2"):
    """Add the retry features. Returns a copy."""
    from sentence_transformers import SentenceTransformer, util

    d = d.copy()
    # device="cpu": the Mac GPU backend (MPS) crashed on some machines.
    embedder = SentenceTransformer(embedder_name, device="cpu")  # cuts text at 256 tokens
    a = embedder.encode(d.original_request.tolist(), convert_to_tensor=True, show_progress_bar=True)
    b = embedder.encode(d.followup_request.tolist(), convert_to_tensor=True, show_progress_bar=True)
    d["sim_to_original"] = util.cos_sim(a, b).diagonal().cpu().numpy()

    text = d.followup_request
    for name, pattern in PATTERNS.items():
        d[name] = text.str.contains(pattern, case=False, regex=True).astype(int)
    d["is_question"] = text.str.contains(r"\?", regex=True).astype(int)
    clean = lambda x: x.str.lower().str.split().str.join(" ")   # ignore case and spacing
    d["exact_repeat"] = (clean(text) == clean(d.original_request)).astype(int)
    d["log_retry_words"] = np.log1p(text.str.split().str.len())
    d["long_paste"] = (text.str.split().str.len() > 100).astype(int)   # pasted prompt, code or story
    d["log_orig_words"] = np.log1p(d.original_request.str.split().str.len())
    return d


def split_halves(d, seed=42):
    """Random 50/50 split into 'discover' and 'confirm'."""
    d = d.copy()
    d["half"] = np.where(np.random.RandomState(seed).rand(len(d)) < 0.5, "discover", "confirm")
    return d


def odds_table(df, outcome="not_refused", features=None):
    """Logistic regression -> odds ratios. Above 1 = goes with outcome=1."""
    import statsmodels.formula.api as smf

    features = features or FEATURES
    formula = f"{outcome} ~ " + " + ".join(features + CONTROLS)
    res = smf.logit(formula, data=df).fit(disp=0)
    ci = np.exp(res.conf_int())
    table = pd.DataFrame({"odds_ratio": np.exp(res.params), "ci_low": ci[0], "ci_high": ci[1],
                          "p": res.pvalues}).round(3)
    return table, int(res.nobs)


def rate_table(df, outcome="not_refused", features=None):
    """For each feature: its average in each group (for 0/1 features, the share that has it).
    Also the number of messages with the feature in the group where outcome=1."""
    features = features or FEATURES
    means = df.groupby(outcome)[features].mean().T
    table = pd.DataFrame({"group_0": means[0.0], "group_1": means[1.0]})
    table["difference"] = table.group_1 - table.group_0
    binary = [f for f in features if set(df[f].dropna().unique()) <= {0, 1}]
    table["count_in_group_1"] = df[df[outcome] == 1][binary].sum().reindex(table.index)
    return table.round(3)
