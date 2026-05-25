from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_target_distribution(df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="y")
    plt.title("Target Distribution")
    plt.xlabel("Subscribe Term Deposit")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["no", "yes"],
        cmap="Blues",
        values_format="d",
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(roc_data: Dict[str, Tuple], save_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    for name, (y_true, y_proba) in roc_data.items():
        RocCurveDisplay.from_predictions(y_true, y_proba, name=name, ax=ax)
    plt.title("ROC Curves")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_top_features(importance_df: pd.DataFrame, save_path: Path, top_n: int = 20) -> None:
    top_df = importance_df.head(top_n).sort_values(by="importance", ascending=True)
    plt.figure(figsize=(8, 7))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.title(f"Top {top_n} Important Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
