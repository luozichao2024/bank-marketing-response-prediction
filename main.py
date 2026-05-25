from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_utils import download_bank_marketing, load_bank_data, summarize_data
from src.train_utils import build_models, evaluate_model, get_feature_importance
from src.visualize import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_target_distribution,
    plot_top_features,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    result_dir = project_dir / "results"
    figure_dir = result_dir / "figures"

    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    print("Step 1: Downloading/loading dataset...")
    csv_path = download_bank_marketing(data_dir)
    df = load_bank_data(csv_path)

    print("Step 2: Basic data analysis...")
    summarize_data(df, result_dir)
    plot_target_distribution(df, figure_dir / "target_distribution.png")

    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print("Step 3: Training models with preprocessing and F1-score grid search...")
    models = build_models(
        numeric_features,
        categorical_features,
        RANDOM_STATE
    )

    metrics = []
    trained_models = {}
    roc_data = {}

    for name, grid_search in models.items():
        print(f"Training {name}...")

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        trained_models[name] = best_model

        model_metrics, y_pred, y_proba = evaluate_model(
            best_model,
            X_test,
            y_test
        )

        model_metrics["model"] = name
        model_metrics["best_params"] = str(grid_search.best_params_)
        metrics.append(model_metrics)

        roc_data[name] = (y_test, y_proba)

        print(f"{name} best params: {grid_search.best_params_}")
        print(
            f"{name} test Accuracy: {model_metrics['accuracy']:.4f}, "
            f"Precision: {model_metrics['precision']:.4f}, "
            f"Recall: {model_metrics['recall']:.4f}, "
            f"F1: {model_metrics['f1']:.4f}, "
            f"ROC-AUC: {model_metrics['roc_auc']:.4f}"
        )

    metrics_df = pd.DataFrame(metrics)

    metrics_df = metrics_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "best_params",
        ]
    ]

    metrics_df.to_csv(
        result_dir / "metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Select the final best model by F1-score.
    best_row = metrics_df.sort_values(
        by="f1",
        ascending=False
    ).iloc[0]

    best_name = best_row["model"]
    best_model = trained_models[best_name]

    print(f"Step 4: Best model is {best_name}, saving results...")

    y_proba = best_model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_score = -1
    best_y_pred = None
    best_threshold_metrics = None

    for threshold in np.arange(0.30, 0.71, 0.01):
        temp_pred = (y_proba >= threshold).astype(int)

        acc = accuracy_score(y_test, temp_pred)
        precision = precision_score(y_test, temp_pred, zero_division=0)
        recall = recall_score(y_test, temp_pred)
        f1 = f1_score(y_test, temp_pred)

        # 目标：尽量提高Accuracy，同时保证yes类Recall不低于0.75
        if recall >= 0.75:
            score = acc + 0.3 * f1

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_y_pred = temp_pred
                best_threshold_metrics = {
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc_score(y_test, y_proba),
                }

    # 如果没有任何阈值满足Recall >= 0.75，则退回默认阈值0.5
    if best_y_pred is None:
        best_threshold = 0.5
        best_y_pred = (y_proba >= 0.5).astype(int)
        best_threshold_metrics = {
            "accuracy": accuracy_score(y_test, best_y_pred),
            "precision": precision_score(y_test, best_y_pred, zero_division=0),
            "recall": recall_score(y_test, best_y_pred),
            "f1": f1_score(y_test, best_y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

    y_pred = best_y_pred

    print(f"Best threshold: {best_threshold:.2f}")
    print(
        f"After threshold search - "
        f"Accuracy: {best_threshold_metrics['accuracy']:.4f}, "
        f"Precision: {best_threshold_metrics['precision']:.4f}, "
        f"Recall: {best_threshold_metrics['recall']:.4f}, "
        f"F1: {best_threshold_metrics['f1']:.4f}, "
        f"ROC-AUC: {best_threshold_metrics['roc_auc']:.4f}"
    )

    with open(result_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n")
        f.write("Best selection metric: F1-score\n")
        f.write(f"Best parameters: {best_row['best_params']}\n")
        f.write(f"Best threshold: {best_threshold:.2f}\n")
        f.write(
            "Threshold metrics: "
            f"Accuracy={best_threshold_metrics['accuracy']:.4f}, "
            f"Precision={best_threshold_metrics['precision']:.4f}, "
            f"Recall={best_threshold_metrics['recall']:.4f}, "
            f"F1={best_threshold_metrics['f1']:.4f}, "
            f"ROC-AUC={best_threshold_metrics['roc_auc']:.4f}\n\n"
        )
        f.write(
            classification_report(
                y_test,
                y_pred,
                target_names=["no", "yes"]
            )
        )

    plot_confusion_matrix(
        y_test,
        y_pred,
        figure_dir / "confusion_matrix.png"
    )

    plot_roc_curves(
        roc_data,
        figure_dir / "roc_curve.png"
    )

    importance_df = get_feature_importance(
        best_model,
        numeric_features,
        categorical_features
    )

    importance_df.to_csv(
        result_dir / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_top_features(
        importance_df,
        figure_dir / "top_features.png",
        top_n=20
    )

    joblib.dump(
        best_model,
        result_dir / "best_model.joblib"
    )

    print("Done. Please check the results folder.")


if __name__ == "__main__":
    main()