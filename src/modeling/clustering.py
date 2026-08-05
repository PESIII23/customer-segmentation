"""
Clustering models for customer segmentation (Milestone 2).

Follows the structure of the reference notebooks in docs/reference_code/:
    1. Build RFM ratings (percentile ranks) as the model input
    2. Run four clustering techniques over a parameter sweep
    3. Compare them with cluster-validity metrics (elbow + silhouette)
    4. Band the winning model's clusters into Gold / Silver / Bronze
    5. Profile the value of each band

Usage:
    from src.modeling import clustering
    segments, results, selection = clustering.run_segmentation(rfm, clean_txns)
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MeanShift
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS = PROJECT_ROOT / "docs"
RANDOM_STATE = 42
RATING_COLS = ["RECENCY_RATING", "FREQUENCY_RATING", "MONETARY_RATING"]

_logger = logging.getLogger(__name__)


# ---- model input -----------------------------------------------------------
def build_rfm_ratings(rfm: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Rank customers on each RFM behaviour and rescale to a 0-5 rating.

    Ratings put R, F and M on one comparable scale, so no further standardization
    is needed before distance-based clustering. Recency is ranked descending
    because fewer days since purchase is the better customer.
    """
    df = rfm.copy()
    bins = 5.0
    df["RECENCY_RATING"] = df["RECENCY"].rank(method="min", pct=True, ascending=False) * bins
    df["FREQUENCY_RATING"] = df["FREQUENCY"].rank(method="min", pct=True, ascending=True) * bins
    df["MONETARY_RATING"] = df["MONETARY"].rank(method="min", pct=True, ascending=True) * bins

    # Quartile scores 1-4 give an interpretable RFM_SCORE used to rank the
    # clusters when banding them Gold / Silver / Bronze.
    df["R_SCORE"] = pd.qcut(df["RECENCY"], 4, labels=[4, 3, 2, 1]).astype(int)
    df["F_SCORE"] = pd.qcut(df["FREQUENCY"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    df["M_SCORE"] = pd.qcut(df["MONETARY"], 4, labels=[1, 2, 3, 4]).astype(int)
    df["RFM_SCORE"] = df[["R_SCORE", "F_SCORE", "M_SCORE"]].sum(axis=1)

    # Dummy/proxy variables carried through for segment profiling.
    country = transactions.groupby("CUSTOMER_ID")["COUNTRY"].first()
    df["COUNTRY"] = df["CUSTOMER_ID"].map(country)
    df["IS_UK"] = (df["COUNTRY"] == "United Kingdom").astype(int)

    # Both sides gross: MONETARY is net of refunds, and dividing gross Q4 spend by
    # a near-zero net denominator blows the ratio up for heavily-refunded customers.
    q4_spend = (transactions[transactions["INVOICE_DATE"].dt.month >= 9]
                .groupby("CUSTOMER_ID")["TOTAL_PRICE"].sum())
    denom = df["GROSS_MONETARY"] if "GROSS_MONETARY" in df else df["MONETARY"]
    df["Q4_SPEND_SHARE"] = (df["CUSTOMER_ID"].map(q4_spend).fillna(0) / denom).round(4)
    return df


# ---- evaluation ------------------------------------------------------------
def evaluate(X: pd.DataFrame, labels: np.ndarray) -> dict:
    """Cluster-validity metrics. Undefined for a single cluster, so guard for it."""
    n_clusters = len(set(labels) - {-1})  # -1 is DBSCAN noise
    if n_clusters < 2:
        return {"n_clusters": n_clusters, "silhouette": np.nan,
                "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "n_clusters": n_clusters,
        "silhouette": round(float(silhouette_score(X, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
        "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
    }


# ---- the four modeling iterations ------------------------------------------
def run_kmeans(X: pd.DataFrame, k_values=range(1, 13)) -> pd.DataFrame:
    """Iteration 1 — K-Means across k, capturing SSE for the elbow method."""
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append({"algorithm": "K-Means", "parameter": f"k={k}", "k": k,
                     "sse": round(model.inertia_, 1), **evaluate(X, model.labels_)})
    return pd.DataFrame(rows)


def run_meanshift(X: pd.DataFrame, bandwidths=np.arange(0.5, 1.7, 0.1)) -> pd.DataFrame:
    """Iteration 2 — Mean Shift, which infers cluster count from bandwidth."""
    rows = []
    for bw in bandwidths:
        model = MeanShift(bandwidth=bw).fit(X)
        rows.append({"algorithm": "Mean Shift", "parameter": f"bandwidth={bw:.1f}",
                     "k": np.nan, "sse": np.nan, **evaluate(X, model.labels_)})
    return pd.DataFrame(rows)


def run_dbscan(X: pd.DataFrame, eps_values=np.arange(0.20, 0.40, 0.05),
               min_samples_values=range(5, 50, 10)) -> pd.DataFrame:
    """Iteration 3 — DBSCAN, a density method that also flags noise points."""
    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            rows.append({"algorithm": "DBSCAN",
                         "parameter": f"eps={eps:.2f}, min_samples={min_samples}",
                         "k": np.nan, "sse": np.nan,
                         "noise_points": int((labels == -1).sum()),
                         **evaluate(X, labels)})
    return pd.DataFrame(rows)


def run_agglomerative(X: pd.DataFrame, k_values=range(2, 13)) -> pd.DataFrame:
    """Iteration 4 — Ward hierarchical clustering as a cross-check on K-Means."""
    rows = []
    for k in k_values:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        rows.append({"algorithm": "Agglomerative", "parameter": f"k={k}", "k": k,
                     "sse": np.nan, **evaluate(X, labels)})
    return pd.DataFrame(rows)


# ---- segment banding -------------------------------------------------------
def band_customers(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Rank clusters by mean RFM score: best -> Gold, worst -> Bronze, rest -> Silver.

    With k>3 the middle clusters collapse into Silver, as the reference notebook
    does, so the business always receives exactly three actionable tiers.
    """
    out = df.assign(CLUSTER=labels)
    order = out.groupby("CLUSTER")["RFM_SCORE"].mean().sort_values(ascending=False).index
    bands = {cluster: "Silver" for cluster in order}
    bands[order[0]] = "Gold"
    bands[order[-1]] = "Bronze"
    out["SEGMENT"] = out["CLUSTER"].map(bands)
    return out


def cluster_profile(df: pd.DataFrame, by: str = "SEGMENT") -> pd.DataFrame:
    """Value profile per segment, following the reference notebook's cluster table."""
    g = df.groupby(by)
    profile = pd.DataFrame({
        "CUSTOMERS": g.size(),
        "CUSTOMER_PCT": (100 * g.size() / len(df)).round(1),
        "MEAN_RECENCY": g["RECENCY"].mean().round(1),
        "MEAN_FREQUENCY": g["FREQUENCY"].mean().round(2),
        "MEAN_MONETARY": g["MONETARY"].mean().round(2),
        "MEAN_AOV": g["AVG_ORDER_VALUE"].mean().round(2),
        "MEAN_RFM_SCORE": g["RFM_SCORE"].mean().round(2),
        "UK_PCT": (100 * g["IS_UK"].mean()).round(1),
        "TRANSACTIONS": g["FREQUENCY"].sum(),
        "REVENUE": g["MONETARY"].sum().round(2),
    })
    profile["REVENUE_PCT"] = (100 * profile["REVENUE"] / profile["REVENUE"].sum()).round(1)
    # ARPC / ARPT: average revenue per customer and per transaction.
    profile["ARPC"] = (profile["REVENUE"] / profile["CUSTOMERS"]).round(2)
    profile["ARPT"] = (profile["REVENUE"] / profile["TRANSACTIONS"]).round(2)
    order = ["Gold", "Silver", "Bronze"]
    return profile.reindex(order) if by == "SEGMENT" else profile


# ---- figures ---------------------------------------------------------------
def _subdir(name: str = "modeling") -> Path:
    path = DOCS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_elbow(kmeans_results: pd.DataFrame):
    """Elbow method: SSE against k, with silhouette alongside for the same sweep."""
    path = _subdir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.plot(kmeans_results["k"], kmeans_results["sse"], "b+-", markersize=10)
    ax1.set(title="The Elbow Method", xlabel="Number of clusters (k)",
            ylabel="Sum of squared errors (SSE)")
    scored = kmeans_results.dropna(subset=["silhouette"])
    ax2.plot(scored["k"], scored["silhouette"], "o-", color="#4C72B0")
    ax2.set(title="K-Means Silhouette Score vs k", xlabel="Number of clusters (k)",
            ylabel="Silhouette score")
    plt.tight_layout()
    plt.savefig(path / "elbow_method.png", dpi=110)
    plt.close()


def plot_algorithm_comparison(results: pd.DataFrame):
    """Best silhouette achieved by each technique, and metric detail by cluster count."""
    path = _subdir()
    valid = results.dropna(subset=["silhouette"])
    best = valid.loc[valid.groupby("algorithm")["silhouette"].idxmax()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    ax1.bar(best["algorithm"], best["silhouette"], color="#4C72B0")
    ax1.bar_label(ax1.containers[0], fmt="%.3f", padding=3)
    ax1.set(title="Best Silhouette Score by Technique", ylabel="silhouette")
    ax1.tick_params(axis="x", rotation=15)

    # Cap the axis at 15: a few degenerate runs find 100+ clusters and flatten the rest.
    for algo, grp in valid[valid["n_clusters"] <= 15].groupby("algorithm"):
        grp = grp.sort_values("n_clusters")
        ax2.plot(grp["n_clusters"], grp["silhouette"], marker="o", label=algo, alpha=0.8)
    ax2.set(title="Silhouette vs Number of Clusters (<=15)", xlabel="clusters found",
            ylabel="silhouette")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path / "algorithm_comparison.png", dpi=110)
    plt.close()


def plot_segments(df: pd.DataFrame):
    """Segment scatter on the RFM rating planes, distributions, and value share."""
    path = _subdir()
    palette = {"Gold": "#C8A227", "Silver": "#8C8C8C", "Bronze": "#A05C2B"}
    order = ["Gold", "Silver", "Bronze"]

    # 3D view of the rating space the model actually clustered on.
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    for seg in order:
        s = df[df["SEGMENT"] == seg]
        ax.scatter(s["RECENCY_RATING"], s["FREQUENCY_RATING"], s["MONETARY_RATING"],
                   c=palette[seg], label=seg, s=8, alpha=0.5)
    ax.set(title="Customer Segments in RFM Rating Space",
           xlabel="Recency Rating", ylabel="Frequency Rating", zlabel="Monetary Rating")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path / "segment_scatter_3d.png", dpi=110)
    plt.close()

    pairs = [("RECENCY", "MONETARY"), ("FREQUENCY", "MONETARY"), ("FREQUENCY", "RECENCY")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (x, y) in zip(axes, pairs):
        sns.scatterplot(data=df, x=x, y=y, hue="SEGMENT", hue_order=order, palette=palette,
                        alpha=0.5, s=16, ax=ax, legend=(ax is axes[0]))
        ax.set(title=f"{x} vs {y}", xscale="log", yscale="log")
    plt.tight_layout()
    plt.savefig(path / "segment_scatter.png", dpi=110)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, var in zip(axes, ["RECENCY", "FREQUENCY", "MONETARY"]):
        sns.boxplot(data=df, x="SEGMENT", y=var, hue="SEGMENT", order=order,
                    palette=palette, legend=False, ax=ax)
        ax.set(title=f"{var} by segment", yscale="log")
    plt.tight_layout()
    plt.savefig(path / "segment_boxplots.png", dpi=110)
    plt.close()

    profile = cluster_profile(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = [palette[s] for s in profile.index]
    for ax, col, title in ((ax1, "CUSTOMER_PCT", "Share of customers"),
                           (ax2, "REVENUE_PCT", "Share of revenue")):
        ax.bar(profile.index, profile[col], color=colors)
        ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
        ax.set(title=title, ylabel="%")
    plt.tight_layout()
    plt.savefig(path / "segment_value.png", dpi=110)
    plt.close()


# ---- orchestration ---------------------------------------------------------
def run_segmentation(rfm: pd.DataFrame, transactions: pd.DataFrame,
                     log=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run all four iterations, select the best model, and band Gold/Silver/Bronze.

    Returns (segmented_customers, iteration_results, selection_summary).
    """
    model_input = build_rfm_ratings(rfm, transactions)
    X = model_input[RATING_COLS]

    kmeans_results = run_kmeans(X)
    results = pd.concat([kmeans_results, run_meanshift(X), run_dbscan(X),
                         run_agglomerative(X)], ignore_index=True)
    for algo, grp in results.groupby("algorithm", sort=False):
        best = grp.loc[grp["silhouette"].idxmax()] if grp["silhouette"].notna().any() else None
        if best is not None:
            log(f"      {algo:<14} best silhouette {best['silhouette']:.4f} "
                f"({best['parameter']}, {int(best['n_clusters'])} clusters)")

    # Select on silhouette, restricted to >=3 clusters so the model can express
    # three tiers, and to <=8 so the bands stay interpretable to the business.
    eligible = results[(results["n_clusters"] >= 3) & (results["n_clusters"] <= 8)].dropna(
        subset=["silhouette"])
    best = eligible.loc[eligible["silhouette"].idxmax()]
    log(f"      Selected: {best['algorithm']} ({best['parameter']})")

    # Refit the winning configuration and band every customer.
    k = int(best["n_clusters"])
    if best["algorithm"] == "Agglomerative":
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    else:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
    segmented = band_customers(model_input, labels)

    selection = {
        "algorithm": best["algorithm"],
        "parameter": best["parameter"],
        "n_clusters": k,
        "silhouette": float(best["silhouette"]),
        "calinski_harabasz": float(best["calinski_harabasz"]),
        "davies_bouldin": float(best["davies_bouldin"]),
        "holdout_silhouette": _holdout_check(X, k, log),
    }
    return segmented, results, selection


def _holdout_check(X: pd.DataFrame, k: int, log=print) -> float:
    """Stability check: fit K-Means on 80% of customers, score the unseen 20%.

    Keeps the fit off the holdout rows so the score is not self-graded.
    """
    train, holdout = train_test_split(X, test_size=0.2, random_state=RANDOM_STATE)
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(train)
    score = round(float(silhouette_score(holdout, model.predict(holdout))), 4)
    log(f"      Holdout silhouette (fit on 80%, scored on unseen 20%): {score:.4f}")
    return score
