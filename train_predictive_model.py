from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)
OUT_DIR = Path(__file__).resolve().parent


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def make_dataset(n_samples: int = 700) -> pd.DataFrame:
    """Create a realistic binary outcome dataset for a supervised ML demo."""
    age = RNG.normal(44, 12, n_samples).clip(18, 75)
    monthly_income = RNG.lognormal(mean=8.25, sigma=0.42, size=n_samples)
    debt_ratio = RNG.beta(2.2, 5.0, n_samples)
    credit_score = RNG.normal(680, 65, n_samples).clip(420, 840)
    prior_defaults = RNG.poisson(0.35, n_samples).clip(0, 4)
    engagement_score = RNG.beta(3.0, 2.2, n_samples) * 100

    linear_risk = (
        -3.4
        + 0.035 * (age - 40)
        - 0.00018 * (monthly_income - 3600)
        + 3.6 * debt_ratio
        - 0.013 * (credit_score - 650)
        + 0.95 * prior_defaults
        - 0.018 * (engagement_score - 50)
        + RNG.normal(0, 0.65, n_samples)
    )
    probability = sigmoid(linear_risk)
    outcome = RNG.binomial(1, probability)

    return pd.DataFrame(
        {
            "age": age.round(1),
            "monthly_income": monthly_income.round(2),
            "debt_ratio": debt_ratio.round(3),
            "credit_score": credit_score.round(0).astype(int),
            "prior_defaults": prior_defaults.astype(int),
            "engagement_score": engagement_score.round(1),
            "outcome_default_risk": outcome.astype(int),
        }
    )


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.25):
    indices = RNG.permutation(len(y))
    test_count = int(round(len(y) * test_size))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1
    return (train - mean) / std, (test - mean) / std


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.08, epochs: int = 2500, l2: float = 0.03):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xb = np.c_[np.ones(len(X)), X]
        weights = np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            pred = sigmoid(Xb @ weights)
            gradient = (Xb.T @ (pred - y)) / len(y)
            gradient[1:] += self.l2 * weights[1:] / len(y)
            weights -= self.learning_rate * gradient
        self.weights = weights
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fitted.")
        return sigmoid(np.c_[np.ones(len(X)), X] @ self.weights)


@dataclass
class Node:
    prediction: float
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 20, max_features: int | None = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root: Node | None = None
        self.feature_importances_: np.ndarray | None = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 1.0 - p**2 - (1.0 - p) ** 2

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        parent_gini = self._gini(y)
        best_gain, best_feature, best_threshold = 0.0, None, None
        feature_count = X.shape[1]
        features = np.arange(feature_count)
        if self.max_features:
            features = RNG.choice(features, size=min(self.max_features, feature_count), replace=False)

        for feature in features:
            values = np.unique(np.percentile(X[:, feature], [10, 20, 30, 40, 50, 60, 70, 80, 90]))
            for threshold in values:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() < 5 or right_mask.sum() < 5:
                    continue
                weighted_gini = (
                    left_mask.mean() * self._gini(y[left_mask])
                    + right_mask.mean() * self._gini(y[right_mask])
                )
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, int(feature), float(threshold)
        return best_feature, best_threshold, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(prediction=float(y.mean()))
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return node

        feature, threshold, gain = self._best_split(X, y)
        if feature is None or threshold is None or gain <= 1e-8:
            return node

        self.feature_importances_[feature] += gain * len(y)
        mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root = self._build(X, y, 0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _predict_row(self, row: np.ndarray, node: Node) -> float:
        while node.feature is not None and node.threshold is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
            if node is None:
                return 0.5
        return node.prediction

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Model has not been fitted.")
        return np.array([self._predict_row(row, self.root) for row in X])


class RandomForestClassifier:
    def __init__(self, n_estimators: int = 55, max_depth: int = 6):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees: list[DecisionTreeClassifier] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.trees = []
        max_features = max(1, int(math.sqrt(X.shape[1])))
        importances = np.zeros(X.shape[1])
        for _ in range(self.n_estimators):
            idx = RNG.integers(0, len(y), len(y))
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=18, max_features=max_features)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            importances += tree.feature_importances_
        self.feature_importances_ = importances / max(importances.sum(), 1e-12)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([tree.predict_proba(X) for tree in self.trees], axis=0)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def roc_points(y_true: np.ndarray, scores: np.ndarray):
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    points = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        points.append((fpr, tpr))
    points = sorted(set(points))
    auc = float(np.trapezoid([p[1] for p in points], [p[0] for p in points]))
    return points, auc


def markdown_metrics_table(df: pd.DataFrame) -> str:
    columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in df[columns].to_dict("records"):
        cells = []
        for col in columns:
            value = record[col]
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def metrics(y_true: np.ndarray, scores: np.ndarray):
    pred = (scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / len(y_true)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    _, auc = roc_points(y_true, scores)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def write_confusion_svg(cm: np.ndarray, path: Path):
    labels = [["TN", "FP"], ["FN", "TP"]]
    max_value = max(int(cm.max()), 1)
    cells = []
    for r in range(2):
        for c in range(2):
            value = int(cm[r, c])
            intensity = value / max_value
            color = f"rgb({int(235 - 115 * intensity)}, {int(245 - 70 * intensity)}, {int(255 - 25 * intensity)})"
            x, y = 130 + c * 150, 90 + r * 120
            cells.append(
                f'<rect x="{x}" y="{y}" width="150" height="120" fill="{color}" stroke="#243447"/>'
                f'<text x="{x+75}" y="{y+50}" text-anchor="middle" font-size="22" font-weight="700">{labels[r][c]}</text>'
                f'<text x="{x+75}" y="{y+82}" text-anchor="middle" font-size="28">{value}</text>'
            )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="470" height="370" viewBox="0 0 470 370">
<rect width="470" height="370" fill="#ffffff"/>
<text x="235" y="38" text-anchor="middle" font-size="24" font-family="Arial" font-weight="700">Confusion Matrix</text>
<text x="235" y="335" text-anchor="middle" font-size="15" font-family="Arial">Predicted class</text>
<text x="30" y="210" transform="rotate(-90 30 210)" text-anchor="middle" font-size="15" font-family="Arial">Actual class</text>
<g font-family="Arial" fill="#14213d">{''.join(cells)}</g>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_roc_svg(curves: dict[str, tuple[list[tuple[float, float]], float]], path: Path):
    colors = {"Logistic Regression": "#2f80ed", "Decision Tree": "#f2994a", "Random Forest": "#219653"}
    width, height = 640, 460
    left, top, plot_w, plot_h = 70, 45, 500, 340

    def xy(point):
        fpr, tpr = point
        return left + fpr * plot_w, top + (1 - tpr) * plot_h

    polylines = []
    legend = []
    for i, (name, (points, auc)) in enumerate(curves.items()):
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, points))
        polylines.append(f'<polyline points="{coords}" fill="none" stroke="{colors[name]}" stroke-width="3"/>')
        ly = 75 + i * 25
        legend.append(f'<line x1="400" y1="{ly}" x2="432" y2="{ly}" stroke="{colors[name]}" stroke-width="4"/>')
        legend.append(f'<text x="440" y="{ly+5}" font-size="14">{name} AUC={auc:.3f}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#ffffff"/>
<text x="{width/2}" y="28" text-anchor="middle" font-size="24" font-family="Arial" font-weight="700">ROC Curve</text>
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="#243447"/>
<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top}" stroke="#8a94a6" stroke-dasharray="6 6"/>
{''.join(polylines)}
<g font-family="Arial" fill="#14213d">
<text x="{left+plot_w/2}" y="430" text-anchor="middle" font-size="15">False Positive Rate</text>
<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" font-size="15">True Positive Rate</text>
<text x="{left-8}" y="{top+plot_h+5}" text-anchor="end" font-size="12">0.0</text>
<text x="{left-8}" y="{top+5}" text-anchor="end" font-size="12">1.0</text>
<text x="{left}" y="{top+plot_h+22}" text-anchor="middle" font-size="12">0.0</text>
<text x="{left+plot_w}" y="{top+plot_h+22}" text-anchor="middle" font-size="12">1.0</text>
{''.join(legend)}
</g>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_importance_svg(names: list[str], values: np.ndarray, path: Path):
    width, height = 720, 410
    left, top, bar_w, gap = 210, 60, 420, 42
    max_value = max(float(values.max()), 1e-12)
    bars = []
    order = np.argsort(values)[::-1]
    for rank, idx in enumerate(order):
        y = top + rank * gap
        w = (values[idx] / max_value) * bar_w
        bars.append(f'<text x="{left-12}" y="{y+19}" text-anchor="end" font-size="14">{names[idx]}</text>')
        bars.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="24" fill="#219653"/>')
        bars.append(f'<text x="{left+w+8}" y="{y+18}" font-size="13">{values[idx]:.3f}</text>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#ffffff"/>
<text x="{width/2}" y="30" text-anchor="middle" font-size="24" font-family="Arial" font-weight="700">Random Forest Feature Importance</text>
<g font-family="Arial" fill="#14213d">{''.join(bars)}</g>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def main():
    data = make_dataset()
    data.to_csv(OUT_DIR / "customer_default_dataset.csv", index=False)

    target = "outcome_default_risk"
    feature_names = [col for col in data.columns if col != target]
    X = data[feature_names].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train_std, X_test_std = standardize(X_train, X_test)

    models = {
        "Logistic Regression": LogisticRegressionGD().fit(X_train_std, y_train),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=22).fit(X_train, y_train),
        "Random Forest": RandomForestClassifier(n_estimators=60, max_depth=6).fit(X_train, y_train),
    }

    rows = []
    roc_curves = {}
    scores_by_name = {}
    for name, model in models.items():
        scores = model.predict_proba(X_test_std if name == "Logistic Regression" else X_test)
        scores_by_name[name] = scores
        row = {"model": name, **metrics(y_test, scores)}
        rows.append(row)
        roc_curves[name] = roc_points(y_test, scores)

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(OUT_DIR / "model_metrics_summary.csv", index=False, float_format="%.4f")

    best_name = metrics_df.iloc[0]["model"]
    best_scores = scores_by_name[best_name]
    best_cm = confusion_matrix(y_test, (best_scores >= 0.5).astype(int))
    write_confusion_svg(best_cm, OUT_DIR / "confusion_matrix.svg")
    write_roc_svg(roc_curves, OUT_DIR / "roc_curve.svg")
    write_importance_svg(feature_names, models["Random Forest"].feature_importances_, OUT_DIR / "feature_importance.svg")

    report = f"""# Predictive Modeling Using Machine Learning

This project trains and evaluates supervised models that predict whether a customer is at default risk from tabular customer attributes.

## Dataset

- Rows: {len(data)}
- Features: {len(feature_names)}
- Target: `{target}` where `1` means higher default risk and `0` means lower default risk
- Train/test split: 75% / 25%

## Models Compared

- Logistic Regression with gradient descent
- Decision Tree using Gini impurity
- Random Forest with bootstrapped decision trees

## Results

{markdown_metrics_table(metrics_df)}

Best model by ROC AUC: **{best_name}**

## Visualizations

- `confusion_matrix.svg` shows classification counts for the best model at a 0.50 threshold.
- `roc_curve.svg` compares model discrimination across all thresholds.
- `feature_importance.svg` shows which fields the Random Forest relied on most.

## How To Run

```powershell
& 'C:\\Users\\hp\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe' .\\train_predictive_model.py
```
"""
    (OUT_DIR / "README.md").write_text(report, encoding="utf-8")

    print(metrics_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    print(f"Best model: {best_name}")
    print(f"Artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
