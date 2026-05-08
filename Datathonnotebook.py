# =============================================================================
# WILDFIRE EVACUATION-RISK PREDICTION  —  ANNOTATED TEACHING VERSION
# =============================================================================
# Competition: WiDS Worldwide Globalthon 2026
#
# THE PROBLEM
# -----------
# For each fire event, we want to estimate the probability that the fire will
# "hit" (e.g., reach a populated boundary) within 12, 24, 48, and 72 hours.
# This is a MULTI-HORIZON BINARY CLASSIFICATION problem: four related yes/no
# questions, ordered in time. A natural constraint follows: if we are 80%
# sure it hits within 24h, we must be at least 80% sure it hits within 48h.
# We will enforce this monotonicity at the very end.
#
# THE PIPELINE (high level)
# -------------------------
#   1.  Load data
#   2.  Build the four binary targets from `event` and `time_to_hit_hours`
#   3.  Audit feature quality (missingness, near-constant columns, redundancy)
#   4.  Try several feature-pruning variants
#   5.  Add a few engineered features (logs, interactions, thresholds)
#   6.  Compare models (RandomForest / CatBoost / XGBoost) across horizons
#   7.  Use feature importance to prune further
#   8.  Tune CatBoost hyperparameters with class weighting
#   9.  Diagnose errors and CALIBRATION on a validation split
#  10.  Final model: 5-fold × 3-repeat CV ensemble of all three models
#  11.  Enforce horizon monotonicity, write submission.csv
# =============================================================================


# -----------------------------------------------------------------------------
# 1. IMPORTS AND DATA DISCOVERY
# -----------------------------------------------------------------------------
# We rely on three gradient/forest libraries because each has different
# inductive biases — averaging them tends to be more robust than any one alone.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Walk the Kaggle input directory and print every file we find.
# This is a sanity check — never assume the data path; verify it.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
# -----------------------------------------------------------------------------
# 2. ROBUST DATA LOADING
# -----------------------------------------------------------------------------
# Kaggle sometimes mounts data at slightly different paths. Instead of hard-
# coding one location, we glob for any train.csv / test.csv under /kaggle/input
# and fall back to the canonical path if globbing fails. This makes the same
# notebook work in re-runs, forks, and offline environments.
import glob

train_candidates = glob.glob("/kaggle/input/**/train.csv", recursive=True)
test_candidates = glob.glob("/kaggle/input/**/test.csv", recursive=True)

if train_candidates and test_candidates:
    train_path = train_candidates[0]
    test_path = test_candidates[0]
else:
    # Fallback to known path
    train_path = "/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26/train.csv"
    test_path = "/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26/test.csv"

print(f"Train path: {train_path}")
print(f"Test path: {test_path}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# %%
# -----------------------------------------------------------------------------
# 3. BUILD THE TARGETS — one binary label per horizon
# -----------------------------------------------------------------------------
# The raw label tells us:
#   - `event`: did this fire eventually hit at all?  (0/1)
#   - `time_to_hit_hours`: if it hit, how many hours until it did?
#
# A fire is "positive" at horizon h hours iff it hit AND it hit within h hours.
# Notice that hit_by_12h ⊆ hit_by_24h ⊆ hit_by_48h ⊆ hit_by_72h by construction
# (a fire that hits within 12h also hits within 72h). This nesting is the
# reason horizon monotonicity must hold for any sensible predictor.
HORIZONS = [12, 24, 48, 72]
TARGET_COLS = [f"hit_by_{h}h" for h in HORIZONS]

# Columns we never want as features: the ID, the raw event flag, the
# time-to-hit (which would leak the answer), and the targets themselves.
NON_FEATURE_COLS = ["event_id", "event", "time_to_hit_hours"] + TARGET_COLS

for h in HORIZONS:
    hit_by_h = (train["event"] == 1) & (train["time_to_hit_hours"] <= h)
    train[f"hit_by_{h}h"] = hit_by_h.astype(int)

# Always print class balance — heavy imbalance changes which metric and
# which loss function make sense (e.g., AUC > accuracy when positives are rare).
print("Positive label counts by horizon:")
for h in HORIZONS:
    col = f"hit_by_{h}h"
    print(f"  {col}: {train[col].sum()} / {len(train)}")

print(f"\nEvent distribution:\n{train['event'].value_counts()}")

# %%
# -----------------------------------------------------------------------------
# 4. BASE FEATURE MATRIX
# -----------------------------------------------------------------------------
# Drop ID/target/leakage columns; everything else is a candidate feature.
# We keep this as the "base" set and will derive pruned/engineered variants.
X_base = train.drop(columns=NON_FEATURE_COLS).copy()
print("Base feature shape:", X_base.shape)
print("Features:", X_base.columns.tolist())

# %%
# -----------------------------------------------------------------------------
# 5. FEATURE QUALITY AUDIT
# -----------------------------------------------------------------------------
# Before modeling, we ALWAYS profile every column. The three failure modes
# we want to catch:
#   (a) high missingness  — model may overweight whatever imputation we use
#   (b) constant / near-constant — carries no information, just noise
#   (c) (later) redundancy — pairs of features so correlated they double-count
#
# A column where one value occupies, say, 99% of rows is "near-constant" and
# usually safe to drop. The 0.99 threshold is a starting point, not a law.
def build_feature_quality_report(X_df, feature_set_name, near_constant_threshold=0.99):
    rows = []
    n_rows = len(X_df)

    for col in X_df.columns:
        s = X_df[col]
        missing_count = s.isna().sum()
        missing_pct = missing_count / n_rows
        non_null = s.dropna()
        n_unique = non_null.nunique()

        # "top_value" / "top_freq_pct" describe the dominant value:
        # how concentrated is the column around a single value?
        if len(non_null) == 0:
            top_value, top_freq_pct = np.nan, np.nan
        else:
            vc = non_null.value_counts(dropna=True)
            top_value = vc.index[0]
            top_freq_pct = vc.iloc[0] / len(non_null)

        is_constant = n_unique <= 1
        is_near_constant = (
            pd.notna(top_freq_pct)
            and top_freq_pct >= near_constant_threshold
            and not is_constant
        )

        # Bucket findings into human-readable flags so a quick glance is enough.
        flags = []
        if missing_pct >= 0.50:
            flags.append("high_missing")
        elif missing_pct >= 0.20:
            flags.append("medium_missing")
        if is_constant:
            flags.append("constant")
        elif is_near_constant:
            flags.append("near_constant")

        rows.append({
            "feature_set": feature_set_name,
            "feature": col,
            "dtype": str(s.dtype),
            "missing_pct": round(missing_pct, 3),
            "n_unique": n_unique,
            "top_value": top_value,
            "top_freq_pct": round(top_freq_pct, 3) if pd.notna(top_freq_pct) else np.nan,
            "quality_flag": ", ".join(flags) if flags else "ok",
        })

    return pd.DataFrame(rows)


feature_quality_df = build_feature_quality_report(X_base, "base")
print("Feature quality report:")
display(feature_quality_df.sort_values("top_freq_pct", ascending=False).head(15))

# Anything not flagged "ok" deserves a second look.
problems = feature_quality_df[feature_quality_df["quality_flag"] != "ok"]
print(f"\nProblematic features: {len(problems)}")
if len(problems):
    display(problems)

# %%
# -----------------------------------------------------------------------------
# 6. UNIVARIATE SIGNAL — Pearson correlation with the 48h target
# -----------------------------------------------------------------------------
# Correlation only sees LINEAR relationships, so it under-counts features that
# matter through interactions or thresholds. Treat this as a quick screen, not
# a verdict — a tree model can still exploit a feature with corr ≈ 0.
H = 48
target_col = f"hit_by_{H}h"

df_corr = X_base.copy()
df_corr[target_col] = train[target_col].values
corr_series = df_corr.corr(numeric_only=True)[target_col].drop(target_col)
# Sort by ABSOLUTE correlation: a strong negative is just as useful as a strong positive.
corr_sorted = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)

print(f"Top 15 correlations with {target_col}:")
print(corr_sorted.head(15))

# Visual: green = positive correlation with hitting, red = negative.
corr_top = corr_sorted.head(20).sort_values()
colors = ["green" if v > 0 else "red" for v in corr_top.values]
plt.barh(corr_top.index, corr_top.values, color=colors)
plt.axvline(0, color="black")
plt.title(f"Correlation with {target_col}")
plt.xlabel("Pearson Correlation")
plt.tight_layout()
plt.show()

# %%
# -----------------------------------------------------------------------------
# 7. CORRELATION HEATMAP among the most predictive features
# -----------------------------------------------------------------------------
# This shows redundancy among the strongest features. Bright red/blue blocks
# off-diagonal mean two features carry similar information — keeping both
# can hurt linear models and inflate importance scores in tree models.
top_features = corr_sorted.abs().head(12).index.tolist()
corr_matrix_top = X_base[top_features].corr()

plt.figure(figsize=(7, 6))
im = plt.imshow(corr_matrix_top, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
plt.title(f"Feature Correlation Heatmap (Top 12 by |corr with {target_col}|)")
plt.xticks(range(len(top_features)), top_features, rotation=90)
plt.yticks(range(len(top_features)), top_features)
plt.tight_layout()
plt.show()

# %%
# -----------------------------------------------------------------------------
# 8. REDUNDANCY DETECTION — pairs with |corr| > 0.85
# -----------------------------------------------------------------------------
# We use the upper triangle of the correlation matrix to avoid listing each
# pair twice (and to skip the diagonal of 1.0 self-correlations).
# 0.85 is a conservative threshold — pairs above it are usually capturing the
# same underlying signal and one of the two can typically be dropped.
corr_matrix = X_base.corr(numeric_only=True)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
threshold = 0.85

high_corr_pairs = [
    (col, row, upper.loc[row, col])
    for col in upper.columns
    for row in upper.index
    if pd.notnull(upper.loc[row, col]) and abs(upper.loc[row, col]) > threshold
]

print(f"Highly correlated pairs (|corr| > {threshold}):")
for f1, f2, val in high_corr_pairs:
    print(f"  {f1}  <->  {f2}   corr = {val:.3f}")

# Plot only the involved features so the heatmap stays readable.
high_corr_features = list(set([f for pair in high_corr_pairs for f in pair[:2]]))
if len(high_corr_features) > 1:
    corr_subset = corr_matrix.loc[high_corr_features, high_corr_features]
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_subset, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.title("Highly Correlated Feature Subset")
    plt.xticks(range(len(high_corr_features)), high_corr_features, rotation=90)
    plt.yticks(range(len(high_corr_features)), high_corr_features)
    plt.tight_layout()
    plt.show()

# %%
# -----------------------------------------------------------------------------
# 9. THREE FEATURE-PRUNING VARIANTS
# -----------------------------------------------------------------------------
# Rather than commit to one pruning rule, we build three candidate matrices
# and let the validation set decide which one models prefer:
#
#   A. base_full                  — keep everything (control)
#   B. base_pruned_dominant_90pct — drop columns whose dominant value is ≥ 90%
#                                   of rows (less aggressive than 99%)
#   C. base_pruned_dominant_0     — manually drop motion features known to be
#                                   mostly zero (domain knowledge override)
#
# Lesson: pruning is hypothesis-driven; always test rather than declare.
X_base_full = X_base.copy()

drop_dominant_90pct = feature_quality_df[
    feature_quality_df["top_freq_pct"] >= 0.90
]["feature"].tolist()
X_base_pruned_dominant_90pct = X_base.drop(columns=drop_dominant_90pct, errors="ignore").copy()

manual_drop = [
    "closing_speed_m_per_h",
    "closing_speed_abs_m_per_h",
    "projected_advance_m",
    "dist_std_ci_0_5h",
    "dist_fit_r2_0_5h",
]
X_base_pruned_dominant_0 = X_base.drop(columns=manual_drop, errors="ignore").copy()

pruning_feature_sets = {
    "base_full": X_base_full,
    "base_pruned_dominant_90pct": X_base_pruned_dominant_90pct,
    "base_pruned_dominant_0": X_base_pruned_dominant_0,
}

for name, df in pruning_feature_sets.items():
    print(f"{name}: {df.shape}")

print("\nDropped in 90pct:", sorted(set(X_base.columns) - set(X_base_pruned_dominant_90pct.columns)))
print("Dropped in manual:", sorted(set(X_base.columns) - set(X_base_pruned_dominant_0.columns)))

# %%
# -----------------------------------------------------------------------------
# 10. FEATURE ENGINEERING
# -----------------------------------------------------------------------------
# Tree models can split anywhere, so they don't *need* engineered features —
# but well-chosen transformations often help by reducing the number of splits
# required to discover a relationship. We add:
#
#   * log_dist_min        — distance is right-skewed; log compresses the tail
#   * dist_close / very_close — explicit threshold flags at meaningful cutoffs
#   * dist_x_area         — interaction: a CLOSE AND LARGE fire is worse than
#                           either alone
#   * dist_per_area       — ratio: distance normalized by fire size
#   * log_dist_x_log_area — interaction in log space if available
#   * dist_x_slope        — does the fire's distance trend matter at threshold?
#
# Wrap this in a function so we can apply EXACTLY THE SAME transforms to both
# train and test. (Doing it twice by hand is the #1 source of train/test skew.)
def add_engineered_features(df):
    """Add engineered features to a dataframe. Works for both train and test."""
    out = df.copy()

    # log1p = log(1 + x): handles zeros gracefully and tames the long tail.
    out["log_dist_min"] = np.log1p(out["dist_min_ci_0_5h"])

    # Threshold flags help models split cleanly at known meaningful distances.
    out["dist_close"] = (out["dist_min_ci_0_5h"] < 5000).astype(int)
    out["dist_very_close"] = (out["dist_min_ci_0_5h"] < 1000).astype(int)

    # Multiplicative interaction: small distance × large area = big risk.
    out["dist_x_area"] = out["dist_min_ci_0_5h"] * out["area_first_ha"]

    # +1 in the denominator avoids divide-by-zero for tiny fires.
    out["dist_per_area"] = out["dist_min_ci_0_5h"] / (out["area_first_ha"] + 1)

    # Optional features — only build them if the inputs exist in this df.
    if "log1p_area_first" in out.columns:
        out["log_dist_x_log_area"] = out["log_dist_min"] * out["log1p_area_first"]

    if "dist_slope_ci_0_5h" in out.columns:
        out["dist_x_slope"] = out["dist_min_ci_0_5h"] * out["dist_slope_ci_0_5h"]

    return out

# Apply to the manually-pruned set (our domain-informed favorite).
X_eng = add_engineered_features(X_base_pruned_dominant_0)

# Register the engineered version so it's compared alongside the others.
pruning_feature_sets["base_pruned_dominant_0_eng"] = X_eng

print(f"Engineered feature set: {X_eng.shape}")
print(f"New features: {sorted(set(X_eng.columns) - set(X_base_pruned_dominant_0.columns))}")

# %%
# -----------------------------------------------------------------------------
# 11. MODEL ZOO + EVALUATION HARNESS
# -----------------------------------------------------------------------------
# We build small FACTORY FUNCTIONS rather than instances so each cross-
# validation fold gets a fresh, untrained model. Why three different models?
#
#   * RandomForest — bagging of deep trees; very robust, hard to overfit,
#                    low capacity for very subtle interactions.
#   * CatBoost     — gradient boosting with native categorical handling and
#                    strong defaults; usually our single best model here.
#   * XGBoost      — gradient boosting with `hist` tree method for speed;
#                    different regularization profile than CatBoost so the
#                    ensemble of the two captures more.
#
# Hyperparameter notes (these were lightly tuned, not optimal):
#   - n_estimators is set high; we'll let early stopping pick the real number
#     during the tuning section. For this comparison we fit to completion.
#   - subsample/colsample_bytree below 1.0 inject randomness → less overfit.

def rf_model_factory():
    return RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42
    )

def cb_model_factory():
    return CatBoostClassifier(
        iterations=2000, learning_rate=0.03, depth=6,
        l2_leaf_reg=5.0, subsample=0.8, random_seed=42,
        loss_function="Logloss", eval_metric="AUC",
        verbose=False, allow_writing_files=False
    )

def xgb_model_factory():
    return XGBClassifier(
        n_estimators=3000, learning_rate=0.02, max_depth=5,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.0, reg_lambda=1.0, gamma=0.0,
        eval_metric="logloss", tree_method="hist",
        random_state=42, n_jobs=-1
    )

model_dict = {
    "RandomForest": rf_model_factory,
    "CatBoost": cb_model_factory,
    "XGBoost": xgb_model_factory,
}

# -----------------------------------------------------------------------------
# Evaluation function
# -----------------------------------------------------------------------------
# Trains one model per horizon on the same split and returns the metrics.
#
# Two metrics, on purpose:
#   * AUC  — RANKING quality. "If I pick a positive and a negative at random,
#            how often is the positive scored higher?" Insensitive to
#            calibration. 0.5 = random, 1.0 = perfect ranking.
#   * Brier — squared error of the probability vs the 0/1 label. Penalizes
#             miscalibration. Lower is better. AUC alone can hide a model
#             that ranks well but outputs nonsense probabilities.
def evaluate_model_across_horizons(
    model_factory, model_name, feature_set_name,
    X_df, train_df, idx_train, idx_valid, horizons
):
    results = []
    trained_models = {}
    X_tr = X_df.loc[idx_train]
    X_va = X_df.loc[idx_valid]

    for h in horizons:
        target_col = f"hit_by_{h}h"
        y_tr = train_df.loc[idx_train, target_col]
        y_va = train_df.loc[idx_valid, target_col]

        model = model_factory()
        model.fit(X_tr, y_tr)

        prob_va = model.predict_proba(X_va)[:, 1]
        # AUC requires both classes to be present in the validation set.
        auc = roc_auc_score(y_va, prob_va) if len(np.unique(y_va)) > 1 else np.nan
        brier = brier_score_loss(y_va, prob_va)

        results.append({
            "feature_set": feature_set_name,
            "model": model_name,
            "horizon": h,
            "AUC": auc,
            "Brier": brier,
        })
        trained_models[h] = model

    return pd.DataFrame(results), trained_models

# -----------------------------------------------------------------------------
# Stratified train/validation split
# -----------------------------------------------------------------------------
# We stratify on the 48h target so the validation set has the SAME positive
# rate as training. With heavy class imbalance, an unstratified split can land
# all positives in one side and break AUC measurement.
idx_train, idx_valid = train_test_split(
    X_base.index, test_size=0.2, random_state=42,
    stratify=train["hit_by_48h"]
)

print(f"Training rows: {len(idx_train)}")
print(f"Validation rows: {len(idx_valid)}")
print(f"Validation positive rate (48h): {train.loc[idx_valid, 'hit_by_48h'].mean():.2f}")

# %%
# -----------------------------------------------------------------------------
# 12. THE BIG GRID: every (feature set) × (model) × (horizon)
# -----------------------------------------------------------------------------
# This is the workhorse comparison. We keep ALL trained models in
# `all_trained_models` because we'll mine them later for feature importances.
comparison_tables = []
all_trained_models = {}

for feature_set_name, X_df in pruning_feature_sets.items():
    for model_name, factory in model_dict.items():
        result_df, trained_models = evaluate_model_across_horizons(
            model_factory=factory,
            model_name=model_name,
            feature_set_name=feature_set_name,
            X_df=X_df,
            train_df=train,
            idx_train=idx_train,
            idx_valid=idx_valid,
            horizons=HORIZONS,
        )
        comparison_tables.append(result_df)
        all_trained_models[(feature_set_name, model_name)] = trained_models

model_comparison_df = pd.concat(comparison_tables, ignore_index=True)

# %%
# -----------------------------------------------------------------------------
# 13. IMPORTANCE-BASED PRUNING (using CatBoost on the full base set)
# -----------------------------------------------------------------------------
# Idea: use the model itself to tell us which features are weakest, then
# retrain without them. This often beats correlation-based pruning because
# trees see non-linear and interaction signal.
#
# We average importance ACROSS HORIZONS — a feature that matters at 12h but
# not at 72h is still valuable, so we keep it if any horizon liked it.

import pandas as pd
import numpy as np

catboost_base_models = all_trained_models[("base_full", "CatBoost")]

importance_rows = []

for h in HORIZONS:
    model = catboost_base_models[h]
    importances = model.get_feature_importance()

    tmp = pd.DataFrame({
        "feature": X_base.columns,
        "importance": importances,
        "horizon": h
    })
    importance_rows.append(tmp)

catboost_importance_df = pd.concat(importance_rows, ignore_index=True)

display(catboost_importance_df.head(2))

# Aggregate per feature: mean (overall signal), max (peak utility),
# min (worst horizon — useful for spotting unstable features).
importance_summary = (
    catboost_importance_df
    .groupby("feature", as_index=False)
    .agg(
        mean_importance=("importance", "mean"),
        max_importance=("importance", "max"),
        min_importance=("importance", "min")
    )
    .sort_values("mean_importance", ascending=True)  # weakest first
    .reset_index(drop=True)
)

# Show the bottom of the list — these are pruning candidates.
lowest_k = 8
lowest_importance_features = importance_summary.head(lowest_k)["feature"].tolist()

print(f"Lowest-importance features ({len(lowest_importance_features)}): " +
      ", ".join(lowest_importance_features))

display(importance_summary.head(10))

# Build three pruned variants at different aggressiveness levels and let
# validation pick the best. Pruning too few = no benefit; pruning too many =
# you remove useful signal that just happened to rank low.
drop_low_5 = importance_summary.head(5)["feature"].tolist()
drop_low_8 = importance_summary.head(8)["feature"].tolist()
drop_low_10 = importance_summary.head(10)["feature"].tolist()

X_base_imp_pruned_5 = X_base.drop(columns=drop_low_5, errors="ignore").copy()
X_base_imp_pruned_8 = X_base.drop(columns=drop_low_8, errors="ignore").copy()
X_base_imp_pruned_10 = X_base.drop(columns=drop_low_10, errors="ignore").copy()

importance_pruning_feature_sets = {
    "base_full": X_base.copy(),
    "base_imp_pruned_5": X_base_imp_pruned_5,
    "base_imp_pruned_8": X_base_imp_pruned_8,
    "base_imp_pruned_10": X_base_imp_pruned_10,
}

for name, df in importance_pruning_feature_sets.items():
    print(f"{name}: {df.shape}")

print("Dropped in base_imp_pruned_5:")
print(", ".join(drop_low_5))

print("\nDropped in base_imp_pruned_8:")
print(", ".join(drop_low_8))

print("\nDropped in base_imp_pruned_10:")
print(", ".join(drop_low_10))



# %%
# -----------------------------------------------------------------------------
# 14. EVALUATE the importance-pruned variants on CatBoost only
# -----------------------------------------------------------------------------
# We keep the model fixed (CatBoost) so the only variable is the feature set.
# That's the discipline of A/B testing: change one thing at a time.
importance_pruning_results = []
importance_pruning_models = {}

for feature_set_name, X_df in importance_pruning_feature_sets.items():
    result_df, trained_models = evaluate_model_across_horizons(
        model_factory=model_dict["CatBoost"],
        model_name="CatBoost",
        feature_set_name=feature_set_name,
        X_df=X_df,
        train_df=train,
        idx_train=idx_train,
        idx_valid=idx_valid,
        horizons=HORIZONS,
    )

    importance_pruning_results.append(result_df)
    importance_pruning_models[(feature_set_name, "CatBoost")] = trained_models

importance_pruning_comparison_df = pd.concat(importance_pruning_results, ignore_index=True)

# Pivot to make horizon-by-horizon comparison easy to scan visually.
auc_pivot_importance = importance_pruning_comparison_df.pivot(
    index="feature_set",
    columns="horizon",
    values="AUC"
)

print("CatBoost AUC comparison — importance pruning")
display(auc_pivot_importance)

brier_pivot_importance = importance_pruning_comparison_df.pivot(
    index="feature_set",
    columns="horizon",
    values="Brier"
)

print("CatBoost Brier comparison — importance pruning")
display(brier_pivot_importance)


# %%
# -----------------------------------------------------------------------------
# 15. CATBOOST HYPERPARAMETER TRIALS (with class weights + early stopping)
# -----------------------------------------------------------------------------
# A small manual grid — depth, learning rate, L2 regularization. Instead of an
# exhaustive sweep we hand-pick the most informative axes.
#
# Two important upgrades over the earlier comparison:
#
#   1. CLASS WEIGHTS  — positives are rare. We compute w_c = N / (2 * n_c) so
#      the average weight is ~1 but each class contributes equally. This nudges
#      the model to actually care about positives instead of always predicting 0.
#
#   2. EARLY STOPPING — we hand the validation set to CatBoost and let it
#      stop training once AUC plateaus. `iterations=5000` is just an upper
#      bound; `use_best_model=True` rewinds to the best epoch.
#
# `rsm=0.8` and `min_data_in_leaf=50` add regularization (subsample features;
# refuse splits that would isolate fewer than 50 rows).
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import pandas as pd
import numpy as np

from collections import Counter

results = []

# Use the 90%-pruned set here (best signal-to-noise during early experiments).
X = X_base_pruned_dominant_90pct.copy()

X_train = X.loc[idx_train]
X_valid = X.loc[idx_valid]

param_grid = [
    {"depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 3, "subsample": 0.8},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3, "subsample": 0.8},
    {"depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 3, "subsample": 0.8},
    {"depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 5, "subsample": 0.8},
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5, "subsample": 0.8},
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 8, "subsample": 0.8},
]



results = []

for h in HORIZONS:
    target_col = f"hit_by_{h}h"

    target_col = f"hit_by_{h}h"

    y_train = train.loc[idx_train, target_col]
    y_valid = train.loc[idx_valid, target_col]

    # ---- Compute balanced class weights ----
    # Formula: w_c = N / (2 * n_c). Equal-cost classes regardless of rarity.
    counter = Counter(y_train)
    total = len(y_train)

    class_weights = {
        0: total / (2 * counter[0]),
        1: total / (2 * counter[1])
    }

    for i, params in enumerate(param_grid, 1):
        print(f"\nTrial {i}")
        for key, value in params.items():
            print(f"  {key}: {value}")
        model = CatBoostClassifier(
            iterations=5000,
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            subsample=params["subsample"],

            rsm=0.8,                 # column subsampling per tree
            min_data_in_leaf=50,     # don't split into tiny groups (anti-overfit)

            class_weights=class_weights,

            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,

            #**params
        )

        # Pass eval_set so CatBoost can early-stop. `use_best_model` rewinds
        # to the iteration with the best validation AUC.
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=500
        )

        pred_valid = model.predict_proba(X_valid)[:, 1]

        auc = roc_auc_score(y_valid, pred_valid)
        brier = brier_score_loss(y_valid, pred_valid)

        results.append({
            "horizon": h,
            "trial": i,
            **params,
            "best_iteration": model.get_best_iteration(),
            "auc": roc_auc_score(y_valid, pred_valid),
            "brier": brier_score_loss(y_valid, pred_valid)
        })

# Sort: per horizon, best AUC first, then lowest Brier as a tiebreaker.
leaderboard = pd.DataFrame(results).sort_values(
    ["horizon", "auc", "brier"],
    ascending=[True, False, True]
).reset_index(drop=True)

leaderboard



# %%
# -----------------------------------------------------------------------------
# 16. TIDY THE COMPARISON TABLE FOR VISUALIZATION
# -----------------------------------------------------------------------------
# Build a single label "Model (feature_set)" so each line in upcoming plots
# is uniquely identified. Pivots make horizon-by-horizon comparison readable.

model_comparison_df["model_feature_set"] = (
    model_comparison_df["model"] + " (" + model_comparison_df["feature_set"] + ")"
)

# Order rows so the table groups by feature set, which makes patterns obvious.
ordered_rows = [
    f"{model_name} ({feature_set_name})"
    for feature_set_name in pruning_feature_sets.keys()
    for model_name in model_dict.keys()
]

ordered_horizons = list(HORIZONS)


# Pivot tables for comparison

model_comparison_df["label"] = (
    model_comparison_df["model"] + " (" + model_comparison_df["feature_set"] + ")"
)

ordered_rows = [
    f"{m} ({fs})"
    for fs in pruning_feature_sets
    for m in model_dict
]

auc_pivot = model_comparison_df.pivot(
    index="label", columns="horizon", values="AUC"
).reindex(index=ordered_rows, columns=HORIZONS)

brier_pivot = model_comparison_df.pivot(
    index="label", columns="horizon", values="Brier"
).reindex(index=ordered_rows, columns=HORIZONS)

print("AUC comparison")
display(auc_pivot.round(6))

print("\nBrier comparison")
display(brier_pivot.round(6))

# %%
# -----------------------------------------------------------------------------
# 17. PLOT METRICS ACROSS HORIZONS
# -----------------------------------------------------------------------------
# We expect AUC to go DOWN as the horizon grows (predicting 72h ahead is
# inherently noisier than predicting 12h). Brier behavior depends on base rate.
# Lines that drop the slowest are the most "future-aware" models.
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))

for label in ordered_rows:
    df_part = model_comparison_df[
        model_comparison_df["model_feature_set"] == label
    ].sort_values("horizon")

    plt.plot(
        df_part["horizon"],
        df_part["AUC"],
        marker="o",
        label=label
    )

plt.xlabel("Prediction Horizon (hours)")
plt.ylabel("AUC")
plt.title("AUC Across Horizons\nAll Models and Feature Sets")
plt.xticks(ordered_horizons)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Brier
plt.figure(figsize=(9, 5))

for label in ordered_rows:
    df_part = model_comparison_df[
        model_comparison_df["model_feature_set"] == label
    ].sort_values("horizon")

    plt.plot(
        df_part["horizon"],
        df_part["Brier"],
        marker="o",
        label=label
    )

plt.xlabel("Prediction Horizon (hours)")
plt.ylabel("Brier Score")
plt.title("Brier Score Across Horizons\nAll Models and Feature Sets")
plt.xticks(ordered_horizons)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# %%
# -----------------------------------------------------------------------------
# 18. PICK THE BEST MODEL/FEATURE-SET COMBINATION
# -----------------------------------------------------------------------------
# Default to (CatBoost, base_pruned_dominant_0). Then check whether the
# engineered-feature variant did better at the 48h horizon — if so, switch.
# (We anchor on 48h because it's the "primary" horizon for this competition.)
BEST_FS = "base_pruned_dominant_0"
BEST_MODEL = "CatBoost"
X_best = pruning_feature_sets[BEST_FS]
best_models = all_trained_models[(BEST_FS, BEST_MODEL)]
feat_cols = X_best.columns

# Conditional upgrade to engineered features if they helped.
if ("base_pruned_dominant_0_eng", "CatBoost") in all_trained_models:
    eng_auc = model_comparison_df[
        (model_comparison_df["model"] == "CatBoost") &
        (model_comparison_df["feature_set"] == "base_pruned_dominant_0_eng") &
        (model_comparison_df["horizon"] == 48)
    ]["AUC"].values
    base_auc = model_comparison_df[
        (model_comparison_df["model"] == "CatBoost") &
        (model_comparison_df["feature_set"] == "base_pruned_dominant_0") &
        (model_comparison_df["horizon"] == 48)
    ]["AUC"].values
    if len(eng_auc) and len(base_auc) and eng_auc[0] >= base_auc[0]:
        BEST_FS = "base_pruned_dominant_0_eng"
        X_best = pruning_feature_sets[BEST_FS]
        best_models = all_trained_models[(BEST_FS, BEST_MODEL)]
        feat_cols = X_best.columns
        print(f"Engineered features are better! AUC: {eng_auc[0]:.4f} vs {base_auc[0]:.4f}")
    else:
        print(f"Base features are better. Using {BEST_FS}")

print(f"Best feature set: {BEST_FS} ({len(feat_cols)} features)")

# Inspect what the best 48h model thinks is important.
model_48 = best_models[48]
fi_48 = pd.DataFrame({
    "feature": feat_cols,
    "importance": model_48.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nTop 15 features (48h):")
display(fi_48.head(15))

# %%
# -----------------------------------------------------------------------------
# 19. CORRELATION vs MODEL-IMPORTANCE PLOT
# -----------------------------------------------------------------------------
# A scatter where each point is a feature: x = linear correlation with target,
# y = how much the model used it. Interesting points:
#   - high importance, low correlation → the model exploits NON-LINEAR signal
#   - low importance, high correlation → likely redundant with a stronger feature
try:
    corr_48 = X_best.corrwith(train["hit_by_48h"])
    corr_importance_df = pd.DataFrame({
        "feature": feat_cols,
        "correlation": corr_48.values,
        "importance": fi_48.set_index("feature").loc[feat_cols]["importance"].values
    })
    plt.figure(figsize=(8, 6))
    plt.scatter(corr_importance_df["correlation"], corr_importance_df["importance"], alpha=0.7)
    plt.axvline(0, linestyle="--", color="gray")
    plt.xlabel("Correlation with hit_by_48h")
    plt.ylabel("CatBoost Feature Importance")
    plt.title("Feature Correlation vs Model Importance")
    plt.show()
except Exception as e:
    print(f"Correlation plot skipped: {e}")

# %%
# -----------------------------------------------------------------------------
# 20. FEATURE IMPORTANCE PER HORIZON
# -----------------------------------------------------------------------------
# Different horizons may rely on different signals (e.g., short-term distance
# matters more for 12h; longer-term spread for 72h). This view confirms or
# refutes that intuition.
all_fi = []
for h in HORIZONS:
    temp = pd.DataFrame({
        "feature": feat_cols,
        "importance": best_models[h].get_feature_importance(),
        "horizon": f"{h}h"
    })
    all_fi.append(temp)

all_fi_df = pd.concat(all_fi, ignore_index=True)

for h in HORIZONS:
    print(f"\nTop 10 features for hit_by_{h}h")
    display(
        all_fi_df[all_fi_df["horizon"] == f"{h}h"]
        .sort_values("importance", ascending=False)
        .head(10)
    )

# %%
# -----------------------------------------------------------------------------
# 21. ERROR ANALYSIS — build a frame with predictions next to features
# -----------------------------------------------------------------------------
# We'll dissect which rows the model gets wrong and look for patterns.
# `error = 1` means the prediction (using a 0.5 threshold) disagrees with truth.
try:
    h = 48
    target_col = f"hit_by_{h}h"
    y_valid = train.loc[idx_valid, target_col].copy()

    X_valid_best = X_best.loc[idx_valid]
    prob_valid_48 = best_models[h].predict_proba(X_valid_best)[:, 1]
    pred_valid_48 = (prob_valid_48 >= 0.5).astype(int)

    valid_analysis = X_valid_best.copy()
    valid_analysis["y_true"] = y_valid.values
    valid_analysis["prob"] = prob_valid_48
    valid_analysis["y_pred"] = pred_valid_48
    valid_analysis["error"] = (valid_analysis["y_true"] != valid_analysis["y_pred"]).astype(int)
    print(f"Validation analysis built: {valid_analysis.shape}")
except Exception as e:
    print(f"Error analysis skipped: {e}")
    valid_analysis = None

# %%
# -----------------------------------------------------------------------------
# 22. CONFUSION MATRIX (manual count)
# -----------------------------------------------------------------------------
# TP/FP/FN/TN at threshold = 0.5. With imbalanced data, accuracy alone is
# misleading — a model that always predicts 0 can look "accurate" but be
# useless. Always look at the breakdown.
if valid_analysis is not None:
    tp = ((valid_analysis["y_true"] == 1) & (valid_analysis["y_pred"] == 1)).sum()
    tn = ((valid_analysis["y_true"] == 0) & (valid_analysis["y_pred"] == 0)).sum()
    fp = ((valid_analysis["y_true"] == 0) & (valid_analysis["y_pred"] == 1)).sum()
    fn = ((valid_analysis["y_true"] == 1) & (valid_analysis["y_pred"] == 0)).sum()
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"Accuracy: {(tp + tn) / len(valid_analysis):.3f}")
else:
    print("Skipped (no valid_analysis)")

# %%
# -----------------------------------------------------------------------------
# 23. WORST FALSE POSITIVES AND FALSE NEGATIVES
# -----------------------------------------------------------------------------
# Sort FPs by descending probability — these are rows the model was MOST
# confident about being positive but turned out negative. Sort FNs ascending —
# rows the model was most confident were negative but were actually positive.
# Eyeballing these often reveals a missing feature or a labeling oddity.
if valid_analysis is not None:
    false_positives = valid_analysis[
        (valid_analysis["y_true"] == 0) & (valid_analysis["y_pred"] == 1)
    ].sort_values("prob", ascending=False)

    false_negatives = valid_analysis[
        (valid_analysis["y_true"] == 1) & (valid_analysis["y_pred"] == 0)
    ].sort_values("prob", ascending=True)

    print(f"False positives: {len(false_positives)}")
    display(false_positives.head(10))
    print(f"\nFalse negatives: {len(false_negatives)}")
    display(false_negatives.head(10))
else:
    print("Skipped")

# %%
# -----------------------------------------------------------------------------
# 24. FEATURE-LEVEL DIFFERENCES BETWEEN CORRECT AND INCORRECT PREDICTIONS
# -----------------------------------------------------------------------------
# Compare mean feature values for correct vs erroneous rows. Features with the
# largest gap are the ones most associated with the model's confusion — they
# may be where engineering or extra signal could help.
if valid_analysis is not None:
    error_summary = valid_analysis.groupby("error")[feat_cols].mean().T
    if error_summary.shape[1] == 2:
        error_summary.columns = ["Correct_avg", "Error_avg"]
        error_summary["abs_diff"] = (error_summary["Correct_avg"] - error_summary["Error_avg"]).abs()
        display(error_summary.sort_values("abs_diff", ascending=False).head(20))
    else:
        print("No misclassified samples - model got all validation rows correct.")
else:
    print("Skipped")

# %%
# -----------------------------------------------------------------------------
# 25. CALIBRATION CHECK
# -----------------------------------------------------------------------------
# Bucket predictions into 10 probability bins and ask: "of rows we predicted
# at ~30%, what fraction were ACTUALLY positive?" A well-calibrated model
# would land on the diagonal y = x. Big departures mean the model ranks well
# but its probabilities are over- or under-confident — a prime candidate for
# post-hoc calibration (Platt / isotonic) before submission.
if valid_analysis is not None:
    valid_analysis["prob_bucket"] = pd.cut(
        valid_analysis["prob"], bins=np.linspace(0, 1, 11), include_lowest=True
    )
    bucket_summary = valid_analysis.groupby("prob_bucket", observed=False).agg(
        n=("y_true", "size"),
        actual_rate=("y_true", "mean"),
        predicted_avg=("prob", "mean"),
        error_rate=("error", "mean")
    ).reset_index()
    display(bucket_summary)

    plt.figure(figsize=(8, 5))
    plt.plot(bucket_summary["predicted_avg"], bucket_summary["actual_rate"], marker="o", label="Observed")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Average predicted probability")
    plt.ylabel("Actual positive rate")
    plt.title("Calibration check for hit_by_48h")
    plt.legend()
    plt.show()
else:
    print("Skipped")

# %%
# -----------------------------------------------------------------------------
# 26. FINAL MODEL: REPEATED STRATIFIED K-FOLD ENSEMBLE
# -----------------------------------------------------------------------------
# This is the "production" pass. Three new ideas come together:
#
#   (a) REPEATED K-FOLD: 5 folds × 3 different seed shuffles = 15 train/test
#       splits per horizon. More splits → less variance in the test prediction.
#
#   (b) OUT-OF-FOLD (OOF) PREDICTIONS: each training row gets predictions ONLY
#       from folds where it was in the validation set. OOF AUC is the most
#       honest estimate of generalization without a true holdout.
#
#   (c) WEIGHTED ENSEMBLE: CatBoost (0.6) + RandomForest (0.25) + XGBoost (0.15).
#       Weights reflect single-model performance from earlier comparison.
#       Averaging probabilities is one of the most reliable ways to lift AUC.
#
# Each test prediction is averaged over (folds × repeats × models), giving a
# very stable final score.

# Use engineered features for both train and test — apply the SAME function
# to avoid skew. We restrict test columns to those used in training first.
X_final = X_eng.copy()
X_test_final = add_engineered_features(
    test[X_base_pruned_dominant_0.columns].copy()
)
feat_cols_final = X_final.columns

# CV settings
N_SPLITS = 5
N_REPEATS = 3
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

# Per-model weights for the soft-vote ensemble. Sum should be 1.0.
ENSEMBLE = {
    "CatBoost": (cb_model_factory, 0.6),
    "RandomForest": (rf_model_factory, 0.25),
    "XGBoost": (xgb_model_factory, 0.15),
}

# Initialize accumulators: one prediction array per horizon for the test set,
# and per-horizon OOF score holders for the training set.
test_preds = {h: np.zeros(len(test)) for h in HORIZONS}
oof_scores = {h: [] for h in HORIZONS}

for h in HORIZONS:
    target_col = f"hit_by_{h}h"
    y = train[target_col]
    oof_preds = np.zeros(len(train))
    oof_counts = np.zeros(len(train))   # how many times each row was in a val fold

    for fold_idx, (tr_idx, va_idx) in enumerate(rskf.split(X_final, y)):
        X_tr = X_final.iloc[tr_idx]
        X_va = X_final.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        fold_test_pred = np.zeros(len(test))

        # Train each ensemble member on the same fold and combine probabilities.
        for model_name, (factory, weight) in ENSEMBLE.items():
            model = factory()
            model.fit(X_tr, y_tr)

            # Accumulate weighted OOF probabilities for this fold's val rows.
            oof_preds[va_idx] += model.predict_proba(X_va)[:, 1] * weight
            # Accumulate weighted test predictions for averaging later.
            fold_test_pred += model.predict_proba(X_test_final)[:, 1] * weight

        oof_counts[va_idx] += 1
        # Divide by total folds × repeats so the test prediction averages cleanly.
        test_preds[h] += fold_test_pred / (N_SPLITS * N_REPEATS)

    # Each row was in `N_REPEATS` validation folds (once per repeat). Divide
    # to get the average OOF probability per training row.
    oof_preds /= oof_counts
    auc = roc_auc_score(y, oof_preds)
    brier = brier_score_loss(y, oof_preds)
    oof_scores[h] = {"AUC": auc, "Brier": brier}
    print(f"Horizon {h}h - CV AUC: {auc:.4f}, CV Brier: {brier:.4f}")

prob_12h = test_preds[12]
prob_24h = test_preds[24]
prob_48h = test_preds[48]
prob_72h = test_preds[72]

print("\nExample predictions (first 5, 48h):", prob_48h[:5])

# %%
# -----------------------------------------------------------------------------
# 27. ENFORCE HORIZON MONOTONICITY
# -----------------------------------------------------------------------------
# By the definition of the targets, P(hit by 12h) ≤ P(hit by 24h) ≤ ... ≤ P(72h).
# The ensemble doesn't know this constraint, so its raw probabilities can
# violate it (e.g., predict 0.6 for 24h but 0.55 for 48h — impossible).
# We fix this with a running max across horizons. Cheap, exact, never hurts AUC
# of the longer horizons and almost always helps Brier.
prob_12h_fixed = prob_12h
prob_24h_fixed = np.maximum(prob_24h, prob_12h_fixed)
prob_48h_fixed = np.maximum(prob_48h, prob_24h_fixed)
prob_72h_fixed = np.maximum(prob_72h, prob_48h_fixed)

print("Monotonicity check (all should be True):")
print("  12h <= 24h:", np.all(prob_12h_fixed <= prob_24h_fixed))
print("  24h <= 48h:", np.all(prob_24h_fixed <= prob_48h_fixed))
print("  48h <= 72h:", np.all(prob_48h_fixed <= prob_72h_fixed))

# %%
# -----------------------------------------------------------------------------
# 28. WRITE THE SUBMISSION
# -----------------------------------------------------------------------------
# Always sanity-check before saving:
#   - column names match the competition's expected schema
#   - no NaN values (a single NaN can fail the entire upload)
#   - file is actually written to /kaggle/working (the submission directory)
submission = pd.DataFrame({
    "event_id": test["event_id"],
    "prob_12h": prob_12h_fixed,
    "prob_24h": prob_24h_fixed,
    "prob_48h": prob_48h_fixed,
    "prob_72h": prob_72h_fixed,
})

print(submission.head())
print(f"\nMissing values: {submission.isna().any().any()}")

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved: /kaggle/working/submission.csv")
