"""RQ1.3 semantic analysis, following the project proposal.

Unit: one (refused request m_i, candidate response m_j) pair from `matching.build_pairs`.
U_ij = (u_1, ..., u_K) are the non-empty user messages between the refusal and m_j,
oldest first (the `user_messages` column).

For a feature l the proposal defines

    f2(m_i, U_ij, l) = sum_k w_k * phi_l(u_k) - phi_l(m_i),
    w_k = lam^(K-k) / sum_r lam^(K-r),   0 < lam <= 1

so later messages weigh more; lam = 1 is the plain mean.

Two features compare the retry TO the original, so phi_l(m_i) would be a constant
(a message is 100% similar to, and an exact repeat of, itself). For those the
subtracted term is dropped:
    sim_to_original = sum_k w_k * cos(u_k, m_i)
    exact_repeat    = sum_k w_k * [u_k == m_i]
All other features subtract phi_l(m_i), so they measure the CHANGE from the original.

Outcome: `fulfilled` = the LLM judge says the candidate response fully fulfills the
original request. Model: logistic regression with controls for source model, request
category and retry position, standard errors clustered by conversation, and
multiple-testing-adjusted p-values (Holm and Benjamini-Hochberg).
"""
import json

import numpy as np
import pandas as pd

# Word lists for the lexical features (matched case-insensitively).
APOS = r"(?:'|’)?"
PATTERNS = {
    "has_justification": r"\b(?:for (?:a|my) (?:novel|story|book|class|research|thesis|project|game)|fiction(?:al)?|hypothetical(?:ly)?|educational|just curious|i" + APOS + r"m an? (?:nurse|doctor|teacher|student|researcher|writer)|for (?:school|work|safety|awareness))\b",
    "roleplay_frame": r"\b(?:pretend|role.?play|act as|imagine|let" + APOS + r"s say|in a (?:story|fiction)|you are now)\b",
    "pushback": r"\b(?:just do it|do it anyway|try again|stop refusing|ignore (?:the |your |all |any )?(?:previous |above )?(?:instructions|rules|restrictions|policy|policies|guidelines)|no (?:restrictions|rules|limits)|you (?:can|are able to) (?:do|answer|say|tell|write)|jailbreak)\b",
    "polite": r"\b(?:please|kindly|thank)",
}

LEXICAL = ["has_justification", "roleplay_frame", "pushback", "polite", "is_question",
           "log_words", "long_paste"]
RELATIONAL = ["sim_to_original", "exact_repeat"]
FEATURES = RELATIONAL + LEXICAL


# ---------------------------------------------------------------- data
def load_pairs(pairs_path, judgments_path, fulfilled=("full fulfillment",),
               drop=("uncertain",)):
    """Pairs with at least one user message in U_ij, a judge label, and `fulfilled`.

    - retry_number == 0 (a response with no new user message, e.g. a regeneration) is
      dropped: K = 0 leaves the weighted mean undefined.
    - Pairs the judge called 'uncertain' (and any not judged) are dropped.
    - Everything else that is not in `fulfilled` (partial, no fulfillment) is unfulfilled.
    """
    from matching import load_judgments

    p = load_judgments(pairs_path, judgments_path)
    p = p[p.retry_number >= 1]
    p = p[p.label.notna() & ~p.label.isin(drop)].copy()
    p["fulfilled"] = p.label.isin(fulfilled).astype(int)
    p["messages"] = p.user_messages.map(json.loads)
    p["retry_pos"] = p.retry_number.clip(upper=5).astype(str)   # 5 means "5 or more"
    return p.reset_index(drop=True)


# ---------------------------------------------------------------- phi_l
def _clean(s):
    return " ".join(s.lower().split())


def lexical_table(texts):
    """phi_l for every distinct text (one row per text, indexed by the text)."""
    t = pd.Series(sorted(set(texts)), dtype=object)
    out = pd.DataFrame(index=t.values)
    for name, pattern in PATTERNS.items():
        out[name] = t.str.contains(pattern, case=False, regex=True).astype(int).values
    out["is_question"] = t.str.contains(r"\?", regex=True).astype(int).values
    words = t.str.split().str.len().values
    out["log_words"] = np.log1p(words)
    out["long_paste"] = (words > 100).astype(int)
    return out[LEXICAL]


def embed_texts(texts, encode_fn=None, model_name="all-MiniLM-L6-v2"):
    """Unit-length embedding for every distinct text -> dict text -> vector.

    encode_fn(list_of_text) -> array lets you swap the embedder (used for testing).
    The default model cuts text at 256 tokens.
    """
    uniq = sorted(set(texts))
    if encode_fn is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device="cpu")   # MPS crashed on some Macs
        encode_fn = lambda x: model.encode(x, batch_size=64, show_progress_bar=True,
                                           normalize_embeddings=True)
    vecs = np.asarray(encode_fn(uniq), dtype="float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    return dict(zip(uniq, vecs))


def build_cache(pairs, encode_fn=None):
    """Everything that does not depend on lam: phi_l per text and embeddings per text."""
    texts = set(pairs.original_request.fillna(""))
    for msgs in pairs.messages:
        texts.update(msgs)
    texts = sorted(texts)
    return {"lex": lexical_table(texts), "emb": embed_texts(texts, encode_fn)}


# ---------------------------------------------------------------- f2
def weights(K, lam):
    """w_k = lam^(K-k) / sum_r lam^(K-r), k = 1..K (oldest first). Sums to 1."""
    w = lam ** (K - np.arange(1, K + 1))
    return w / w.sum()


def add_f2(pairs, cache, lam=0.5):
    """Add the nine f2 columns for a given lam. Returns a copy."""
    if not 0 < lam <= 1:
        raise ValueError("lam must satisfy 0 < lam <= 1")
    lex, emb = cache["lex"], cache["emb"]
    rows = []
    for orig, msgs in zip(pairs.original_request.fillna(""), pairs.messages):
        w = weights(len(msgs), lam)
        u = lex.loc[msgs, LEXICAL].to_numpy(dtype=float)     # K x features
        o = lex.loc[orig, LEXICAL].to_numpy(dtype=float)
        f2 = list(w @ u - o)                                  # weighted mean minus original
        o_vec = emb[orig]
        sim = np.array([emb[m] @ o_vec for m in msgs])
        same = np.array([_clean(m) == _clean(orig) for m in msgs], dtype=float)
        rows.append([w @ sim, w @ same] + f2)
    out = pairs.copy()
    out[FEATURES] = np.array(rows)
    return out


# ---------------------------------------------------------------- request category
def add_category(pairs, cache, k=8, seed=42):
    """Request category = k-means cluster of the original request's embedding.

    The data has no category label, so this is an approximation. Check the clusters
    with `describe_categories` and rename them if you want.
    """
    from sklearn.cluster import KMeans

    conv = pairs.drop_duplicates("conversation_id")
    X = np.stack([cache["emb"][t] for t in conv.original_request.fillna("")])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    cat = dict(zip(conv.conversation_id, [f"cat{c}" for c in km.labels_]))
    out = pairs.copy()
    out["category"] = out.conversation_id.map(cat)
    return out


def describe_categories(pairs, n=4, width=110):
    """Print a few original requests per category, to name the clusters."""
    conv = pairs.drop_duplicates("conversation_id")
    for c, g in conv.groupby("category"):
        print(f"\n== {c}  ({len(g)} conversations)")
        for t in g.original_request.sample(min(n, len(g)), random_state=0):
            print("  -", " ".join(t.split())[:width])


# ---------------------------------------------------------------- tables
def group_means(df, features=FEATURES, outcome="fulfilled"):
    """Average f2 in unfulfilled (group_0) and fulfilled (group_1) pairs."""
    m = df.groupby(outcome)[features].mean().T
    t = pd.DataFrame({"unfulfilled": m[0], "fulfilled": m[1]})
    t["difference"] = t.fulfilled - t.unfulfilled
    return t.round(3)


def odds_table(df, features=FEATURES, controls=("C(model)", "C(category)", "C(retry_pos)"),
               outcome="fulfilled"):
    """Logistic regression of `outcome` on the f2 features plus controls.

    - Each odds ratio is per 1-unit increase in the f2 value (the raw scale).
    - Standard errors are clustered by conversation (pairs from one conversation share
      their user messages and are not independent).
    - p_holm and p_fdr adjust the feature p-values for testing len(features) features.
    Returns (table, n_pairs, n_conversations).
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests

    d = df.dropna(subset=list(features)).copy()
    formula = f"{outcome} ~ " + " + ".join(list(features) + list(controls))
    groups = pd.factorize(d.conversation_id)[0]
    res = smf.logit(formula, data=d).fit(disp=0, cov_type="cluster",
                                         cov_kwds={"groups": groups})
    ci = np.exp(res.conf_int())
    t = pd.DataFrame({"odds_ratio": np.exp(res.params), "ci_low": ci[0], "ci_high": ci[1],
                      "p": res.pvalues})
    f = t.loc[list(features)].copy()
    f["p_holm"] = multipletests(f.p, method="holm")[1]
    f["p_fdr"] = multipletests(f.p, method="fdr_bh")[1]
    return f.round(3), int(res.nobs), d.conversation_id.nunique()


def lam_sensitivity(pairs, cache, lams=(0.25, 0.5, 1.0), **kw):
    """Odds ratios (and adjusted-significance stars) for each lam, side by side."""
    cols = {}
    for lam in lams:
        t, n, _ = odds_table(add_f2(pairs, cache, lam), **kw)
        cols[f"lam={lam}"] = [f"{r.odds_ratio:.2f}" + ("*" if r.p_holm < 0.05 else "")
                              for r in t.itertuples()]
    out = pd.DataFrame(cols, index=t.index)
    out.attrs["note"] = "* = Holm-adjusted p < 0.05"
    return out
