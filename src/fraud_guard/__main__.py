from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "transactions.csv"
ARTIFACTS = ROOT / "artifacts"
FEATURES = ["amount", "hour", "velocity_1h", "account_age_days", "distance_km", "is_foreign", "device_risk"]


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def generate_transactions(rows: int = 36_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="20min")
    amount = np.clip(rng.lognormal(3.35, 1.0, rows), 1, 2500)
    hour = timestamps.hour.to_numpy()
    velocity = np.clip(rng.poisson(1.3, rows), 0, 12)
    account_age = np.clip(rng.exponential(620, rows), 1, 4000)
    distance = np.clip(rng.exponential(28, rows), 0, 1200)
    is_foreign = rng.binomial(1, 0.08, rows)
    device_risk = np.clip(rng.beta(1.4, 5.5, rows), 0, 1)

    drift_start = int(rows * 0.8)
    amount[drift_start:] *= 1.22
    device_risk[drift_start:] = np.clip(device_risk[drift_start:] + 0.09, 0, 1)
    logit = (
        -7.0 + 0.0038 * amount + 0.62 * velocity + 6.5 * device_risk
        + 1.8 * is_foreign + 0.003 * distance - 0.0006 * account_age
        + 1.0 * ((hour <= 4) | (hour >= 23))
    )
    fraud = rng.binomial(1, sigmoid(logit))
    return pd.DataFrame(
        {
            "transaction_id": [f"TX-{i:06d}" for i in range(rows)],
            "timestamp": timestamps,
            "amount": amount.round(2),
            "hour": hour,
            "velocity_1h": velocity,
            "account_age_days": account_age.round(1),
            "distance_km": distance.round(1),
            "is_foreign": is_foreign,
            "device_risk": device_risk.round(4),
            "is_fraud": fraud,
        }
    )


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = pd.cut(reference, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    cur_counts = pd.cut(current, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    ref = np.clip(ref_counts.to_numpy(), 1e-5, None)
    cur = np.clip(cur_counts.to_numpy(), 1e-5, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def expected_cost(y_true: pd.Series, probability: np.ndarray, threshold: float) -> float:
    predicted = probability >= threshold
    false_negative_loss = ((y_true == 1) & ~predicted).sum() * 240
    review_cost = predicted.sum() * 20
    return float(false_negative_loss + review_cost)


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    data = generate_transactions()
    data.to_csv(DATA_PATH, index=False)

    train = data.iloc[: int(len(data) * 0.60)]
    validation = data.iloc[int(len(data) * 0.60) : int(len(data) * 0.80)]
    production = data.iloc[int(len(data) * 0.80) :].copy()
    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.07, max_leaf_nodes=22, l2_regularization=0.6, random_state=42)
    model.fit(train[FEATURES], train["is_fraud"])
    validation_probability = model.predict_proba(validation[FEATURES])[:, 1]

    thresholds = np.linspace(0.01, 0.85, 170)
    costs = np.array([expected_cost(validation["is_fraud"], validation_probability, value) for value in thresholds])
    threshold = float(thresholds[costs.argmin()])
    production["fraud_probability"] = model.predict_proba(production[FEATURES])[:, 1]
    production["send_to_review"] = production["fraud_probability"] >= threshold

    drift_rows = []
    for feature in FEATURES:
        psi = population_stability_index(train[feature], production[feature])
        drift_rows.append({"feature": feature, "psi": round(psi, 4), "status": "alert" if psi >= 0.2 else "watch" if psi >= 0.1 else "stable"})
    drift = pd.DataFrame(drift_rows).sort_values("psi", ascending=False)
    drift.to_csv(ARTIFACTS / "drift_report.csv", index=False)
    production.to_csv(ARTIFACTS / "production_scores.csv", index=False)
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold}, ARTIFACTS / "model.joblib")

    importance = permutation_importance(model, validation[FEATURES], validation["is_fraud"], scoring="average_precision", n_repeats=4, random_state=42)
    importance_frame = pd.DataFrame({"feature": FEATURES, "importance": importance.importances_mean}).sort_values("importance")
    predictions = production["send_to_review"]
    metrics = {
        "fraud_rate": round(float(production["is_fraud"].mean()), 4),
        "average_precision": round(average_precision_score(production["is_fraud"], production["fraud_probability"]), 4),
        "decision_threshold": round(threshold, 4),
        "precision_at_threshold": round(precision_score(production["is_fraud"], predictions, zero_division=0), 4),
        "recall_at_threshold": round(recall_score(production["is_fraud"], predictions, zero_division=0), 4),
        "review_rate": round(float(predictions.mean()), 4),
        "drift_alerts": int((drift["status"] == "alert").sum()),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(thresholds, costs, color="#2563eb", linewidth=2)
    axes[0].axvline(threshold, color="#f97316", linestyle="--", label=f"Selected: {threshold:.2f}")
    axes[0].set(title="Threshold minimizes expected loss", xlabel="Fraud probability threshold", ylabel="Validation cost (USD)")
    axes[0].legend(frameon=False)
    colors = ["#dc2626" if status == "alert" else "#f59e0b" if status == "watch" else "#0f766e" for status in drift["status"]]
    axes[1].barh(drift["feature"], drift["psi"], color=colors)
    axes[1].axvline(0.2, color="#dc2626", linestyle="--", linewidth=1, label="Alert threshold")
    axes[1].set(title="Production feature drift", xlabel="Population Stability Index")
    axes[1].legend(frameon=False)
    fig.suptitle("Fraud Guard MLOps | Decision and Monitoring", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "fraud_monitoring.png", dpi=160)
    plt.close(fig)
    importance_frame.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

